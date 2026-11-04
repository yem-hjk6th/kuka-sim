"""First PTP of every generated program: start-independent speed bound or declared start pose."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kukasim import first_ptp as fpm
from kukasim import outputs
from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim.krl import NOT_RUN, write_krl
from kukasim.planner import Planner, tcp_poses
from kukasim.transit import TransitPlanner

REPO = Path(__file__).resolve().parent.parent
DOWN = np.array([0, 0, -1.0])


@pytest.fixture(scope="module")
def cell():
    cfg = load_config()
    robot = Robot.from_config(cfg)
    pl = Planner(robot, cfg)
    return cfg, robot, pl


def _tip_and_axes(robot, q):
    """Tool tip (N,3) and, per axis, a point on it and its direction (N,6,3) in the robot root."""
    frames = robot.link_frames(q)
    tip = (frames[-1] @ robot.T_tool)[..., :3, 3]
    pts = np.stack([F[..., :3, 3] for F in frames], 1)
    dirs = np.stack([F[..., :3, :3] @ j.axis for F, j in zip(frames, robot.joints)], 1)
    return tip, pts, dirs


def test_axis_tip_radii_bound_the_sampled_distances(cell):
    """r_i >= the distance of the tool tip from the line of axis i, sampled over the joint ranges."""
    _, robot, _ = cell
    r = fpm.axis_tip_radii(robot)
    rng = np.random.default_rng(11)
    q = rng.uniform(robot.limits[:, 0], robot.limits[:, 1], size=(20000, 6))
    tip, pts, dirs = _tip_and_axes(robot, q)
    v = tip[:, None] - pts
    dist = np.linalg.norm(v - np.sum(v * dirs, -1)[..., None] * dirs, axis=-1)
    assert (dist.max(0) <= r + 1e-9).all(), (dist.max(0), r)
    assert (dist.max(0) >= 0.8 * r).all()           # and not wildly loose (sampled: 84-100 %)
    # the committed numbers (PR description)
    assert np.allclose(r, [1428.6, 1403.6, 843.5, 330.3, 327.0, 225.6], atol=0.1)
    assert fpm.tcp_speed_bound(robot, 1) == pytest.approx(236.9, abs=0.05)


def _sync_ptp_peak(robot, q0, q1, pct=100.0, n=400):
    """Sampled peak TCP speed (mm/s) of synchronous PTPs q0 -> q1 (M,6) at the cruise rate of the
    leading axis at pct % of the URDF axis maxima (every axis at <= its share of its maximum)."""
    dq = q1 - q0
    sdot = np.min(robot.vmax * pct / 100.0 / np.maximum(np.abs(dq), 1e-15), axis=1)     # 1/s
    s = np.linspace(0.0, 1.0, n)
    q = q0[:, None] + s[None, :, None] * dq[:, None]
    tip = (robot.fk_flange(q) @ robot.T_tool)[..., :3, 3]
    dxds = np.linalg.norm(np.diff(tip, axis=1), axis=-1) * (n - 1)
    return (dxds * sdot[:, None]).max(1)


def test_brute_force_random_starts_never_exceed_the_bound(cell):
    """Item 1: random start poses inside the joint limits, synchronous PTP at the URDF axis maxima,
    to random targets and to the approach targets of the generated programs: the sampled peak
    TCP speed never exceeds B(p) = sum_i p % vmax_i r_i (checked at 100 % and at 1 %)."""
    cfg, robot, pl = cell
    rng = np.random.default_rng(5)
    lo, hi = robot.limits[:, 0], robot.limits[:, 1]
    q0 = rng.uniform(lo, hi, size=(3000, 6))
    q1 = rng.uniform(lo, hi, size=(3000, 6))
    # approach targets: the first PTP of a perimeter-like program and of the Hilbert square
    top = cfg.bed()["top_z"]
    T = tcp_poses(np.array([[713, -410, top + 31.0], [690, -400, top + 21.3]]), np.tile(DOWN, (2, 1)),
                  pl.thetas).reshape(-1, 4, 4)
    Q, ok = robot.ik(T, limits=pl.hard_limits)
    targets = Q[ok][:40]
    starts = rng.uniform(lo, hi, size=(len(targets) * 50, 6))
    q0 = np.vstack([q0, starts])
    q1 = np.vstack([q1, np.repeat(targets, 50, axis=0)])
    for pct in (100.0, 1.0):
        peak = _sync_ptp_peak(robot, q0, q1, pct)
        bound = fpm.tcp_speed_bound(robot, pct)
        assert peak.max() <= bound, (pct, peak.max(), bound)
        assert peak.max() >= 0.3 * bound          # the samples do reach a sizeable share of it
    # the bound grows linearly with $VEL_AXIS
    assert fpm.tcp_speed_bound(robot, 7) == pytest.approx(7 * fpm.tcp_speed_bound(robot, 1))


def _two_point_plan(pl, p0, p1, d=DOWN):
    """A plan with two one-point segments (one transit), as in tests/test_transit.py."""
    def state(p):
        T = tcp_poses(p[None], d[None], np.radians([90.0]))[0, 0]
        Q, ok = pl.robot.ik(T, limits=pl.hard_limits)
        ok = ok[0] & (pl.coll.clearance(Q[0]) >= 0)
        return Q[0][np.argmax(ok)]
    q = np.array([state(p0), state(p1)])
    tp = SimpleNamespace(name="first", max_tilt_deg=0.0, points=np.array([p0, p1]), dirs=np.array([d, d]))
    return SimpleNamespace(toolpath=tp, q=q, bins=np.zeros(2, int), feasible=np.ones(2, bool),
                           segments=[(0, 0), (1, 1)], no_state=[])


def _export(cfg, robot, pl, path):
    top = cfg.bed()["top_z"]
    res = _two_point_plan(pl, np.array([600, -450, top + 1.3]), np.array([800, -450, top + 1.3]))
    tpl = TransitPlanner(robot, cfg, pl.coll, pl.hard_limits)
    rep = write_krl(path, robot, res, outputs.tcp_rows(robot, res), cfg, "first", tpl)
    lines = path.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("  PTP"))
    return rep, lines, k


def _vel_before(lines, k):
    vel = {}
    for ln in lines[:k]:
        m = re.match(r"  \$VEL_AXIS\[(\d)\] = (\S+)$", ln)
        if m:
            vel[int(m.group(1))] = float(m.group(2))
    return [vel[i] for i in range(1, 7)]


def test_header_case_bound(cell, tmp_path):
    """Item 2/3, case 'bound': with a TCP rule loose enough that B(1) <= k x max_tcp, the program sets
    $VEL_AXIS = the largest integer p with B(p) <= k x max_tcp before the first PTP, and the header
    says the speed is bounded from any start with the resulting peak."""
    _, robot, pl = cell
    cfg = load_config(planner_overrides={"transit": {"max_tcp_speed_mm_s": 1000}})
    lim = 0.8 * 1000
    b1 = fpm.tcp_speed_bound(robot, 1)
    p = fpm.bound_pct(robot, lim, 100)
    assert p == int(lim // b1) == 3
    assert fpm.tcp_speed_bound(robot, p) <= lim < fpm.tcp_speed_bound(robot, p + 1)
    rep, lines, k = _export(cfg, robot, pl, tmp_path / "bound.src")
    assert lines[0] == NOT_RUN
    assert _vel_before(lines, k) == [p] * 6
    for i in range(1, 7):
        assert f"  $ACC_AXIS[{i}] = {cfg.planner['transit']['acc_axis_pct']:g}" in lines[:k]
    head = "\n".join(ln for ln in lines if ln.startswith(";"))
    assert "case 'bound'" in head and "from ANY start pose" in head
    assert f"Peak TCP speed of the first PTP <= {fpm.tcp_speed_bound(robot, p):.1f} mm/s" in head
    assert "DECLARED START" not in head
    assert rep["first_ptp"]["case"] == "bound" and rep["first_ptp"]["vel_axis_pct"] == p
    # capped at the configured axis speed
    assert fpm.bound_pct(robot, 1e6, 5) == 5


def test_header_case_declared_start(cell, tmp_path):
    """Item 2/3, case 'declared start' (the repository's configuration: B(1) = 236.9 > 80 mm/s): the
    header requires the declared start pose with its tolerance, gives B(1) for reference and the model
    peak; the $VEL_AXIS before the first PTP keeps the model peak from the start (and the tolerance-box
    corners) <= k x max_tcp."""
    cfg, robot, pl = cell
    tp = cfg.planner["transit"]
    lim = tp["speed_safety_factor"] * tp["max_tcp_speed_mm_s"]
    assert fpm.bound_pct(robot, lim) is None
    rep, lines, k = _export(cfg, robot, pl, tmp_path / "declared.src")
    f = rep["first_ptp"]
    assert f["case"] == "declared start"
    assert f["start_axis_deg"] == tp["start_axis"] and f["start_tolerance_deg"] == tp["start_tolerance_deg"]
    assert f["peak_tcp_mm_s"] <= lim and f["peak_tcp_nominal_start_mm_s"] <= f["peak_tcp_mm_s"]
    vel = _vel_before(lines, k)
    assert vel == [f["vel_axis_pct"]] * 6
    head = "\n".join(ln for ln in lines if ln.startswith(";"))
    s = ", ".join(f"A{i + 1} {v:g}" for i, v in enumerate(tp["start_axis"]))
    assert f"THE ROBOT MUST BE AT THE DECLARED START POSE {{{s}}}" in head
    assert f"within +-{tp['start_tolerance_deg']:g} deg on every axis" in head
    assert f"even at $VEL_AXIS 1 % the bound is {fpm.tcp_speed_bound(robot, 1):.1f} mm/s" in head
    assert f"model peak TCP speed of the first PTP {f['peak_tcp_mm_s']:.1f} mm/s" in head
    assert "DECLARED START" in lines[k]
    # re-compute with the transit model from the declared start to the exported target
    qd = np.radians([float(v) for v in re.findall(r"A\d (-?[\d.]+)", lines[k])])
    tpl = TransitPlanner(robot, cfg, pl.coll, pl.hard_limits)
    m = tpl.ptp(np.radians(tp["start_axis"]), qd, pct=vel[0], acc_pct=tp["acc_axis_pct"])
    assert m.peak_tcp_mm_s <= lim + 0.5
    # an invalid declared start is rejected
    bad = load_config(planner_overrides={"transit": {"start_axis": [0, -90, 90, 0, 0, 350]}})
    with pytest.raises(fpm.StartPoseError):
        fpm.plan_first_ptp(robot, bad, qd, 10)


def test_committed_programs_state_the_first_ptp_case():
    """Item 3: every committed program says which case applies and gives the peak of the first PTP."""
    files = sorted((REPO / "results").glob("*/*.src"))
    assert len(files) == 6        # 5 surfaces + the Hilbert corner test
    for f in files:
        t = f.read_text()
        assert t.startswith(NOT_RUN)
        assert re.search(r"First PTP .*: case '(bound|declared start)'", t), f
        assert re.search(r"peak TCP speed of the first PTP (<= )?[\d.]+ mm/s", t, re.I), f

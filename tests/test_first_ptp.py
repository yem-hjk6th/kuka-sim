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


_sync_ptp_peak = fpm.sync_ptp_peak


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
        if pct == 1.0:
            peak_1pct = peak[:3000].max()         # the 3000 random pairs (fpm.random_start_peak)
    # Review item 2: the declared-start case is NEEDED, not only allowed. B(1) > limit alone does not show
    # it (B is an upper bound); the sampled peak at $VEL_AXIS 1 % does: some starts are above the limit.
    tp = cfg.planner["transit"]
    lim = tp["speed_safety_factor"] * tp["max_tcp_speed_mm_s"]
    assert peak_1pct == pytest.approx(fpm.random_start_peak(robot, 1.0))
    assert peak_1pct == pytest.approx(117.6, abs=0.05)       # the number in RESULTS.md and the headers
    assert peak_1pct > lim, (peak_1pct, lim)
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
    assert f["peak_tcp_mm_s"] <= f["sample_limit_mm_s"] == (1 - tp["start_sample_margin"]) * lim
    assert f["peak_tcp_nominal_start_mm_s"] <= f["peak_tcp_mm_s"]
    vel = _vel_before(lines, k)
    assert vel == [f["vel_axis_pct"]] * 6
    head = "\n".join(ln for ln in lines if ln.startswith(";"))
    s = ", ".join(f"A{i + 1} {v:g}" for i, v in enumerate(tp["start_axis"]))
    assert f"THE ROBOT MUST BE AT THE DECLARED START POSE {{{s}}}" in head
    assert f"within +-{tp['start_tolerance_deg']:g} deg on every axis" in head
    assert f"the upper bound\n;   sum_i 1 % x axis max speed x r_i is {fpm.tcp_speed_bound(robot, 1):.1f} mm/s" \
        in head
    assert f"reach {fpm.random_start_peak(robot, 1.0):.1f} mm/s > {lim:g} mm/s (sampled evidence" in head
    assert f"worst sampled over {f['sampled_starts']} starts in the tolerance box, NOT a bound" in head
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
        # review item 1: a sampled peak is worded as a sample, never as the worst over the box
        assert "worst over" not in t and "64 tolerance-box corners" not in t, f
        if "declared start" in t:
            assert re.search(r"worst sampled over \d+ starts in the tolerance box, NOT a bound", t), f
    for doc in (REPO / "RESULTS.md", REPO / "results" / "hilbert_corner" / "report.md"):
        t = doc.read_text()
        assert "worst over the tolerance box" not in t and "so no start-independent bound exists" not in t
    assert "worst sampled over 985 starts in the box, not a bound" in (REPO / "RESULTS.md").read_text()


def test_tolerance_box_is_sampled_inside_not_only_at_the_corners(cell):
    """Review item 1: the 64 corners of the tolerance box are a sample, not a bound. For this target the
    leading axis changes inside the box and interior starts are faster than every corner, so the first
    PTP is evaluated over the 3^6 grid plus random interior points, and the worst sampled peak must stay
    <= (1 - margin) x limit."""
    cfg, robot, _ = cell
    tp = cfg.planner["transit"]
    start, tol = np.array(tp["start_axis"], float), float(tp["start_tolerance_deg"])
    S = fpm.box_starts(start, tol)
    assert S.shape == (3 ** 6 + fpm.BOX_RANDOM, 6) and (S[0] == start).all()
    assert (np.abs(S - start) <= tol + 1e-12).all()
    corner = np.all(np.isclose(np.abs(S - start), tol), axis=1)
    assert corner.sum() == 64
    assert len({tuple(r) for r in np.round(S, 9)}) == len(S)          # no duplicates
    qt = np.radians(start + [0.412, -2.231, -3.393, 4.699, 0.161, -3.841])
    tpl = TransitPlanner(robot, cfg, None, robot.limits)
    step = np.radians(tp["ptp_sample_deg"])

    def peak(q0):
        n = int(np.ceil(np.abs(qt - q0).max() / step)) + 1
        return tpl.ptp_speed(q0 + np.linspace(0, 1, max(n, 2))[:, None] * (qt - q0), 1, tp["acc_axis_pct"])[0]

    pk = np.array([peak(q0) for q0 in np.radians(S)])
    assert pk[~corner].max() > pk[corner].max() + 1.0, (pk[~corner].max(), pk[corner].max())
    fp = fpm.plan_first_ptp(robot, cfg, qt, tp["acc_axis_pct"])
    assert fp.case == fpm.CASE_DECLARED and fp.n_starts == len(S)
    assert fp.peak_tcp_mm_s == pytest.approx(pk.max()) and fp.peak_nominal_mm_s == pytest.approx(pk[0])
    assert fp.ok and fp.peak_tcp_mm_s <= fp.sample_limit_mm_s == pytest.approx(0.9 * fp.limit_mm_s)
    # independent check: random interior starts with another seed stay below the (unreduced) limit
    rnd = np.radians(start + tol * np.random.default_rng(99).uniform(-1, 1, size=(300, 6)))
    assert max(peak(q0) for q0 in rnd) <= fp.limit_mm_s
    # with a margin that leaves no room the same target is rejected, not silently accepted
    tight = load_config(planner_overrides={"transit": {"start_sample_margin": 1 - (pk.max() - 0.5) / fp.limit_mm_s}})
    with pytest.raises(fpm.StartPoseError):
        fpm.plan_first_ptp(robot, tight, qt, tp["acc_axis_pct"])


def test_no_first_ptp_target_is_not_reported_as_checked(cell):
    """Review item 4: a program without a first PTP target states that there is none and that the first
    motion starts from an unknown pose; no "peak 0.0 mm/s <= limit" and the check row fails."""
    from kukasim import hilbert_corner as hc
    cfg, robot, _ = cell
    for c in (cfg, load_config(planner_overrides={"transit": {"max_tcp_speed_mm_s": 1000}})):   # declared / bound
        fp = fpm.plan_first_ptp(robot, c, None, 10)
        assert fp.case == fpm.CASE_NONE and not fp.ok and fp.peak_tcp_mm_s is None
        head = "\n".join(fp.header())
        assert "NO FIRST PTP" in head and "UNKNOWN" in head and "NOT checked" in head
        assert "0.0 mm/s" not in head and "<=" not in head
        s = fp.summary()
        assert s["peak_tcp_mm_s"] is None and s["ok"] is False and "not checked" in s["peak_kind"]
        row = hc.first_ptp_check(fp)
        assert row["ok"] is False and row["value"] == "no first PTP" and "unknown pose" in row["detail"]


def test_invalid_start_pose_is_a_failed_check_in_report_md(tmp_path):
    """Review item 3: a StartPoseError in the Hilbert generator reaches report.md as a failed check
    row; no .src is written and the script exits with 1."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import run_hilbert_corner
    out = tmp_path / "hc"
    out.mkdir()
    (out / "hilbert_corner.src").write_text("stale\n")              # a stale program must not survive
    bad = load_config(planner_overrides={"transit": {"start_axis": [0, -90, 90, 0, 0, 350]}})
    assert run_hilbert_corner.main(["--out", str(out)], cfg=bad) == 1
    assert not (out / "hilbert_corner.src").exists()
    rep = (out / "report.md").read_text()
    row = next(ln for ln in rep.splitlines() if "first PTP peak TCP speed" in ln)
    assert "FAIL" in row.upper() and "outside the joint limits" in row and "no .src written" in row
    assert "hilbert_corner.src" in rep and "NOT written" in rep


def test_export_csv_guard_checks_every_column(tmp_path):
    """Review item 5: export_krl.py replaces trajectory.csv only if every column matches the committed
    file to the number format (A / C modulo 360), not only the joint angles."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from export_krl import csv_differences
    src = REPO / "results" / "dome" / "trajectory.csv"
    rows = src.read_text().splitlines()
    head = rows[0].split(",")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("\n".join(rows) + "\n")

    def changed(col, value):
        r = rows[1].split(",")
        r[head.index(col)] = value
        b.write_text("\n".join([rows[0], ",".join(r)] + rows[2:]) + "\n")
        return csv_differences(a, b)

    assert changed("A1", rows[1].split(",")[head.index("A1")]) == []
    c = float(rows[1].split(",")[head.index("A")])
    assert changed("A", f"{c + 360 if c <= 0 else c - 360:.4f}") == []      # wrap only: format
    assert changed("x", "0.000") and changed("clearance_mm_capped50", "1.00") and changed("ik_slot", "7")
    assert changed("node_cost", "9.00000") and changed("S_unverified", "6")

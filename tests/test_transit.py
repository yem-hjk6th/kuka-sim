"""Transits between printed segments in the KRL export (issue #2)."""
import csv
import re
from pathlib import Path

import numpy as np
import pytest

from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim.krl import NOT_RUN
from kukasim.planner import Planner, tcp_poses
from kukasim.transit import TransitError, TransitPlanner

REPO = Path(__file__).resolve().parent.parent
DOWN = np.array([0, 0, -1.0])


@pytest.fixture(scope="module")
def cell():
    cfg = load_config()
    robot = Robot.from_config(cfg)
    pl = Planner(robot, cfg)
    return cfg, robot, pl, TransitPlanner(robot, cfg, pl.coll, pl.hard_limits)


def _state(pl, p, d, theta_deg=90.0):
    """A collision-free IK state for TCP p, nozzle d (first valid slot)."""
    T = tcp_poses(p[None], d[None], np.radians([theta_deg]))[0, 0]
    Q, ok = pl.robot.ik(T, limits=pl.hard_limits)
    ok = ok[0] & (pl.coll.clearance(Q[0]) >= 0)
    assert ok.any()
    return Q[0][np.argmax(ok)]


def _perimeter_csv():
    """Joint states of the committed perimeter plan (the plan behind issue #2's examples)."""
    with open(REPO / "results" / "perimeter" / "trajectory.csv") as f:
        rows = list(csv.DictReader(f))
    q = np.array([[float(r[f"A{j}"] or "nan") for j in range(1, 7)] for r in rows])
    seg = np.array([int(r["segment"]) for r in rows])
    return np.radians(q), seg


def test_retract_and_approach_along_nozzle_axis(cell):
    cfg, robot, pl, tpl = cell
    top = cfg.bed()["top_z"]
    safe = top + cfg.planner["transit"]["safe_height_mm"]
    for d in (DOWN, np.array([np.sin(np.radians(25)), 0, -np.cos(np.radians(25))])):
        p0 = np.array([713.0, -410.0, top + 1.3])
        p1 = np.array([760.0, -450.0, top + 1.3])
        q0, q1 = _state(pl, p0, d), _state(pl, p1, d, theta_deg=-60.0)
        tr = tpl.plan(q0, p0, d, q1, p1, d)
        r, a = tr.moves[0], tr.moves[-1]
        assert (r.kind, r.label, a.kind, a.label) == ("LIN", "retract", "LIN", "approach")
        # retract starts on the print point and ends at the safe height, moving back along the nozzle
        assert np.allclose(r.tcp[0], p0) and r.tcp[-1][2] == pytest.approx(safe)
        step = r.tcp[-1] - r.tcp[0]
        assert np.allclose(np.cross(step, d), 0, atol=1e-9) and step @ d < 0
        # approach mirrors it and ends exactly on the next start, in the planned configuration
        assert np.allclose(a.tcp[-1], p1) and a.tcp[0][2] == pytest.approx(safe)
        assert np.allclose(a.q[-1], q1) and np.allclose(r.q[0], q0)
        # orientation is held along both LINs
        for m in (r, a):
            R = robot.fk(m.q)[:, :3, :3]
            assert np.abs(R - R[0]).max() < 1e-6
            assert m.continuous
        # the PTP starts and ends at or above the safe height
        assert tr.ptp.tcp[0][2] >= safe - 1e-6 and tr.ptp.tcp[-1][2] >= safe - 1e-6
        # moves are chained: each starts where the previous one ended
        for m0, m1 in zip(tr.moves[:-1], tr.moves[1:]):
            assert np.allclose(m0.q[-1], m1.q[0], atol=1e-9)


def test_min_retract_above_high_segment_end(cell):
    cfg, robot, pl, tpl = cell
    p0 = np.array([713.0, -410.0, cfg.bed()["top_z"] + 60.0])
    p1 = p0 + [30.0, 0, 0]
    tr = tpl.plan(_state(pl, p0, DOWN), p0, DOWN, _state(pl, p1, DOWN), p1, DOWN)
    assert tr.z_mm == pytest.approx(p0[2] + cfg.planner["transit"]["min_retract_mm"])


def test_issue2_direct_ptp_collides_and_transit_is_clear(cell):
    """Issue #2: perimeter.src lines 16 / 308 / 528 were PTPs with wrist flips at print height.
    The transit check must catch the old moves; the new transits must be clear everywhere."""
    cfg, robot, pl, tpl = cell
    q, seg = _perimeter_csv()
    ends = [(np.nonzero(seg == k)[0][-1], np.nonzero(seg == k + 1)[0][0]) for k in range(seg.max())]
    assert ends == [(301, 428), (643, 763)]
    for e, s in ends:
        # the old move: wrist flip (A5 changes sign) at print height
        assert np.sign(q[e, 4]) != np.sign(q[s, 4]) or abs(np.degrees(q[s, 5] - q[e, 5])) > 100
        old = tpl.direct_ptp(q[e], q[s])
        assert old.clearance_mm < 0, (e, s)               # caught: dips into the bed / below it
        assert old.peak_tcp_mm_s > cfg.planner["transit"]["max_tcp_speed_mm_s"]
        pe, ps = robot.fk(q[e])[:3, 3], robot.fk(q[s])[:3, 3]
        tr = tpl.plan(q[e], pe, DOWN, q[s], ps, DOWN)
        # every sample of every move is collision-free; the PTP and the vertical LINs with the tip
        for m in tr.moves:
            c = pl.coll.clearance(m.q, exempt_tip=not m.checked_tip)
            assert (c >= 0).all(), (m.label, c.min())
        p = tr.ptp
        assert p.checked_tip and (pl.coll.clearance(p.q, exempt_tip=False) >= 0).all()
        # dense joint-space sampling of the PTP
        assert np.degrees(np.abs(np.diff(p.q, axis=0))).max() <= cfg.planner["transit"]["ptp_sample_deg"] + 1e-9
        assert np.allclose(p.q, p.q[0] + np.linspace(0, 1, len(p.q))[:, None] * (p.q[-1] - p.q[0]))
        assert not tr.flags


def test_ptp_at_safe_height_rejected_when_it_hits_a_clamp(cell):
    """Transit 0->1 of the perimeter passes the +X/+Y clamp at 30 mm; it must be lifted, not emitted."""
    cfg, robot, pl, tpl = cell
    q, _ = _perimeter_csv()
    pe, ps = robot.fk(q[301])[:3, 3], robot.fk(q[428])[:3, 3]
    tr = tpl.plan(q[301], pe, DOWN, q[428], ps, DOWN)
    assert tr.tried and "clamp_plus_x_plus_y" in tr.tried[0][1]
    assert tr.z_mm > cfg.bed()["top_z"] + cfg.planner["transit"]["safe_height_mm"]
    cfg2 = load_config(planner_overrides={"transit": {"via_heights_mm": []}})
    tpl2 = TransitPlanner(robot, cfg2, pl.coll, pl.hard_limits)
    with pytest.raises(TransitError, match="no collision-free transit"):
        tpl2.plan(q[301], pe, DOWN, q[428], ps, DOWN)


def test_ptp_speed_bound(cell):
    cfg, robot, pl, tpl = cell
    # pure A1 rotation: the TCP moves on a circle about the A1 axis (BASE Z through the origin)
    q0 = np.radians([0, -90, 90, 0, 45, 0])
    q1 = q0 + np.radians([90, 0, 0, 0, 0, 0])
    m = tpl.ptp(q0, q1, pct=10)
    r = np.hypot(*robot.fk(q0)[:2, 3])
    assert m.peak_tcp_mm_s == pytest.approx(r * robot.vmax[0] * 0.10, rel=1e-3)
    assert m.duration_s == pytest.approx(np.pi / 2 / (robot.vmax[0] * 0.10), rel=1e-9)
    # the leading axis sets the rate: adding a slower-moving joint to the move can only slow it down
    q2 = q1 + np.radians([0, 0, 0, 0, 0, 300])
    m2 = tpl.ptp(q0, q2, pct=10)
    assert m2.duration_s >= m.duration_s
    # the peak scales linearly with $VEL_AXIS
    assert tpl.ptp(q0, q1, pct=5).peak_tcp_mm_s == pytest.approx(m.peak_tcp_mm_s / 2, rel=1e-9)


def test_fast_transit_is_flagged(cell):
    cfg, robot, pl, _ = cell
    q, _ = _perimeter_csv()
    pe, ps = robot.fk(q[643])[:3, 3], robot.fk(q[763])[:3, 3]
    fast = load_config(planner_overrides={"transit": {"vel_axis_pct": 100}})
    tr = TransitPlanner(robot, fast, pl.coll, pl.hard_limits).plan(q[643], pe, DOWN, q[763], ps, DOWN)
    assert tr.peak_tcp_mm_s > 100 and any("TCP speed" in f for f in tr.flags)
    near = load_config(planner_overrides={"transit": {"bed_clearance_mm": 40}})
    tr = TransitPlanner(robot, near, pl.coll, pl.hard_limits).plan(q[643], pe, DOWN, q[763], ps, DOWN)
    assert any("from the bed" in f for f in tr.flags)


def test_exported_perimeter_transits(cell):
    """In the committed export, every move between segments starts and ends at or above the safe
    height (except the retract / approach LINs), the axis speed is set, and the header has the flags."""
    cfg, robot, pl, _ = cell
    safe = cfg.bed()["top_z"] + cfg.planner["transit"]["safe_height_mm"]
    lines = (REPO / "results" / "perimeter" / "perimeter.src").read_text().splitlines()
    assert lines[0] == NOT_RUN
    assert any(ln.startswith("; FLAGS:") for ln in lines)
    for i in range(1, 7):
        assert f"  $VEL_AXIS[{i}] = {cfg.planner['transit']['vel_axis_pct']:g}" in lines
        assert f"  $ACC_AXIS[{i}] = {cfg.planner['transit']['acc_axis_pct']:g}" in lines
    n_transit = 0
    for k, ln in enumerate(lines):
        if not ln.startswith("  ; ---- transit"):
            continue
        n_transit += 1
        end = next(j for j in range(k, len(lines)) if lines[j].startswith("  ; segment"))
        block = lines[k + 1:end]
        labels = [b.split()[-1] for b in block if b.startswith("  ; ")]
        assert labels[0] == "retract" and labels[-1] == "approach" and "transit" in labels
        moves = [b for b in block if b.startswith(("  LIN", "  PTP"))]
        # retract target and approach start (= PTP / lower target) are at transit height
        z = [float(re.search(r"Z (-?[\d.]+)", b).group(1)) for b in moves if b.startswith("  LIN")]
        for b in moves[:-1]:
            if b.startswith("  PTP"):
                qd = [float(v) for v in re.findall(r"A\d (-?[\d.]+)", b)]
                assert robot.fk(np.radians(qd))[2, 3] >= safe - 1e-3
        assert all(zz >= safe - 1e-3 for zz in z[:-1])
    assert n_transit == 2

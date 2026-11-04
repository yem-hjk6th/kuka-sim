"""Transits between printed segments in the KRL export (issue #2)."""
import csv
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim import outputs
from kukasim.krl import NOT_RUN, write_krl
from kukasim.planner import Planner, tcp_poses
from kukasim.transit import Move, Transit, TransitError, TransitPlanner

REPO = Path(__file__).resolve().parent.parent
DOWN = np.array([0, 0, -1.0])
PERIMETER_ENDS = [(301, 428), (643, 763)]     # (last point of segment k, first point of segment k + 1)


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
    # the segment end is printed material too, so printed_clearance applies above it as well
    t = cfg.planner["transit"]
    need = p0[2] + max(t["min_retract_mm"], t["printed_clearance_mm"])
    assert need - 1e-9 <= tr.z_mm <= need + 2.0                   # + the PTP sag, if any
    assert tr.ptp.tcp[:, 2].min() >= p0[2] + t["printed_clearance_mm"] - 1e-6
    cfg0 = load_config(planner_overrides={"transit": {"printed_clearance_mm": 0}})
    tr = TransitPlanner(robot, cfg0, pl.coll, pl.hard_limits).plan(
        _state(pl, p0, DOWN), p0, DOWN, _state(pl, p1, DOWN), p1, DOWN)
    assert tr.z_mm == pytest.approx(p0[2] + t["min_retract_mm"])


def test_issue2_direct_ptp_collides_and_transit_is_clear(cell):
    """Issue #2: perimeter.src lines 16 / 308 / 528 were PTPs at print height between segments.
    The transit check must catch the old moves; the new transits must be clear everywhere.
    (Since the deterministic tie-break of P4-2 the committed plan no longer flips the wrist at
    these breaks, but the direct PTP still runs through a clamp and the bed.)"""
    cfg, robot, pl, tpl = cell
    q, seg = _perimeter_csv()
    ends = [(np.nonzero(seg == k)[0][-1], np.nonzero(seg == k + 1)[0][0]) for k in range(seg.max())]
    assert ends == PERIMETER_ENDS
    for e, s in ends:
        # the old move: a PTP straight between the two print-height poses
        assert abs(robot.fk(q[e])[2, 3] - robot.fk(q[s])[2, 3]) < 1e-3
        old = tpl.direct_ptp(q[e], q[s])
        assert old.clearance_mm < 0, (e, s)               # caught: hits a clamp and dips into the bed
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
        assert not tr.flags, tr.flags


def test_ptp_at_safe_height_rejected_when_it_hits_a_clamp(cell):
    """Transit 0->1 of the perimeter passes the +X/+Y clamp at 30 mm; it must be lifted, not emitted."""
    cfg, robot, pl, tpl = cell
    q, _ = _perimeter_csv()
    e, s = PERIMETER_ENDS[0]
    pe, ps = robot.fk(q[e])[:3, 3], robot.fk(q[s])[:3, 3]
    tr = tpl.plan(q[e], pe, DOWN, q[s], ps, DOWN)
    assert tr.tried and "clamp_plus_x_plus_y" in tr.tried[0][1]
    assert tr.z_mm > cfg.bed()["top_z"] + cfg.planner["transit"]["safe_height_mm"]
    cfg2 = load_config(planner_overrides={"transit": {"via_heights_mm": []}})
    tpl2 = TransitPlanner(robot, cfg2, pl.coll, pl.hard_limits)
    with pytest.raises(TransitError, match="no collision-free transit"):
        tpl2.plan(q[e], pe, DOWN, q[s], ps, DOWN)


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


def test_fast_transit_is_slowed_down_or_rejected(cell):
    """P4-1: the 100 mm/s rule is enforced, not only flagged."""
    cfg, robot, pl, _ = cell
    q, _ = _perimeter_csv()
    e, s = PERIMETER_ENDS[1]
    pe, ps = robot.fk(q[e])[:3, 3], robot.fk(q[s])[:3, 3]
    k = cfg.planner["transit"]["speed_safety_factor"]
    vmax = cfg.planner["transit"]["max_tcp_speed_mm_s"]
    # asking for 100 % axis speed: the export lowers $VEL_AXIS until the model peak is <= k * 100 mm/s
    fast = load_config(planner_overrides={"transit": {"vel_axis_pct": 100}})
    tpf = TransitPlanner(robot, fast, pl.coll, pl.hard_limits)
    tr = tpf.plan(q[e], pe, DOWN, q[s], ps, DOWN)
    p = tr.ptp
    assert tpf.ptp_speed(p.q, 100)[0] > vmax                 # it would be too fast as asked
    assert 1 <= p.vel_axis_pct < 100 and p.vel_axis_pct == int(p.vel_axis_pct)
    assert tr.peak_tcp_mm_s <= k * vmax + 1e-9
    assert not any("TCP speed" in f for f in tr.flags)
    # a limit that cannot be met even at 1 %: TransitError, unless explicitly allowed (then flagged)
    tight = {"speed_safety_factor": 0.001}
    tpt = TransitPlanner(robot, load_config(planner_overrides={"transit": tight}), pl.coll, pl.hard_limits)
    with pytest.raises(TransitError, match="allow_fast_transit"):
        tpt.plan(q[e], pe, DOWN, q[s], ps, DOWN)
    allow = load_config(planner_overrides={"transit": tight | {"allow_fast_transit": True}})
    tr = TransitPlanner(robot, allow, pl.coll, pl.hard_limits).plan(q[e], pe, DOWN, q[s], ps, DOWN)
    assert tr.ptp.vel_axis_pct == 1 and any("TCP speed" in f for f in tr.flags)
    near = load_config(planner_overrides={"transit": {"bed_clearance_mm": 40}})
    tr = TransitPlanner(robot, near, pl.coll, pl.hard_limits).plan(q[e], pe, DOWN, q[s], ps, DOWN)
    assert any("from the bed" in f for f in tr.flags)


def _two_point_plan(pl, p0, p1, d=DOWN, name="lateral"):
    """A minimal plan with two one-point segments, so write_krl exports exactly one transit."""
    q = np.array([_state(pl, p0, d), _state(pl, p1, d)])
    tp = SimpleNamespace(name=name, max_tilt_deg=0.0, points=np.array([p0, p1]), dirs=np.array([d, d]))
    return SimpleNamespace(toolpath=tp, q=q, bins=np.zeros(2, int), feasible=np.ones(2, bool),
                           segments=[(0, 0), (1, 1)], no_state=[])


def _ptp_blocks(lines):
    """[(vel_axis_values, ptp_line)] for every transit PTP in an exported .src."""
    out = []
    for k, ln in enumerate(lines):
        if ln == "  ; PTP transit":
            vel = []
            j = k + 1
            while lines[j].startswith("  $VEL_AXIS["):
                vel.append(float(lines[j].split("=")[1]))
                j += 1
            assert lines[j].startswith("  PTP {A1")
            out.append((vel, lines[j]))
    return out


@pytest.mark.parametrize("a, b", [((500, -450), (900, -450)), ((713, -200), (713, -650))])
def test_lateral_transits_export_within_speed_rule(cell, tmp_path, a, b):
    """P4-1: long lateral transits at the default transit height (A3 among the leading axes) are
    exported with a lowered $VEL_AXIS and a model peak <= k * 100 mm/s (= 80)."""
    cfg, robot, pl, tpl = cell
    top = cfg.bed()["top_z"]
    p0, p1 = np.array([*a, top + 1.3]), np.array([*b, top + 1.3])
    res = _two_point_plan(pl, p0, p1)
    rows = outputs.tcp_rows(robot, res)
    f = tmp_path / "lateral.src"
    rep = write_krl(f, robot, res, rows, cfg, "lateral", tpl)
    (t,) = rep["transits"]
    assert t["z_mm"] == pytest.approx(top + cfg.planner["transit"]["safe_height_mm"])   # default transit height
    lim = cfg.planner["transit"]["speed_safety_factor"] * cfg.planner["transit"]["max_tcp_speed_mm_s"]
    assert lim == pytest.approx(80.0)
    assert t["peak_tcp_mm_s_model"] <= lim
    assert t["ptp_vel_axis_pct"] < cfg.planner["transit"]["vel_axis_pct"]       # it had to be lowered
    # A3 is one of the axes that move most (relative to its speed) on these moves
    dq = np.abs(np.radians(t["ptp_joint_change_deg"])) / robot.vmax
    assert dq[2] >= 0.6 * dq.max()
    # the file sets that $VEL_AXIS on all six axes right before the transit PTP, and the
    # re-computed model peak of the exported PTP at that value is within the rule
    lines = f.read_text().splitlines()
    ((vel, ptp),) = _ptp_blocks(lines)
    assert vel == [t["ptp_vel_axis_pct"]] * 6
    qd = np.radians([float(v) for v in re.findall(r"A\d (-?[\d.]+)", ptp)])
    tr = tpl.plan(res.q[0], rows[0][0], DOWN, res.q[1], rows[1][0], DOWN)    # same plan as the export
    assert np.allclose(np.degrees(tr.ptp.q[-1]), np.degrees(qd), atol=1e-3)
    assert tpl.ptp(tr.ptp.q[0], qd, pct=vel[0]).peak_tcp_mm_s <= lim + 0.5     # 4-decimal rounding in the file


def test_short_move_is_not_flagged_at_cruise_speed(cell):
    """P4-7: a move too short to reach cruise is estimated with its (triangular) profile."""
    cfg, robot, pl, tpl = cell
    q0 = np.radians([0, -90, 90, 0, 45, 0])
    long_ = tpl.ptp(q0, q0 + np.radians([90, 0, 0, 0, 0, 0]), pct=10)
    short = tpl.ptp(q0, q0 + np.radians([0.3, 0, 0, 0, 0, 0]), pct=100)
    assert long_.peak_tcp_mm_s == pytest.approx(long_.cruise_tcp_mm_s, rel=1e-6)   # long: cruise reached
    assert short.cruise_tcp_mm_s == pytest.approx(10 * long_.cruise_tcp_mm_s, rel=1e-4)
    # triangle: v_peak = r * sqrt(a * dtheta) for the leading axis
    r = np.hypot(*robot.fk(q0)[:2, 3])
    a = np.radians(cfg.planner["transit"]["axis_acc_deg_s2"][0]) * cfg.planner["transit"]["acc_axis_pct"] / 100
    assert short.peak_tcp_mm_s == pytest.approx(r * np.sqrt(a * np.radians(0.3)), rel=0.02)
    assert short.peak_tcp_mm_s < cfg.planner["transit"]["max_tcp_speed_mm_s"] < short.cruise_tcp_mm_s
    # so a transit made of such a short PTP is neither slowed down nor flagged
    tpl.limit_speed(short)
    assert short.vel_axis_pct == 100
    tr = Transit(0, 1, 0.0, moves=[short])
    assert not any("TCP speed" in f for f in tpl.flags(tr))


def test_transit_clears_material_already_printed(cell):
    """P4-3: the transit height also clears the highest Z printed so far, not only the segment ends."""
    cfg, robot, pl, tpl = cell
    top = cfg.bed()["top_z"]
    p0, p1 = np.array([713.0, -410.0, top + 1.3]), np.array([760.0, -450.0, top + 1.3])
    q0, q1 = _state(pl, p0, DOWN), _state(pl, p1, DOWN)
    low = tpl.plan(q0, p0, DOWN, q1, p1, DOWN)
    assert low.z_mm == pytest.approx(top + cfg.planner["transit"]["safe_height_mm"])
    hi = tpl.plan(q0, p0, DOWN, q1, p1, DOWN, z_printed=top + 40.0)
    need = top + 40.0 + cfg.planner["transit"]["printed_clearance_mm"]
    # the PTP sags between its ends (joint-space line): its lowest TCP sample is what must clear
    assert hi.z_mm >= need - 1e-9
    assert hi.ptp.tcp[:, 2].min() >= need - 1e-6
    assert any("sags" in r for _, r in hi.tried)
    # plan_transits takes the running maximum over every segment printed before the transit
    from kukasim.transit import plan_transits
    pm = np.array([600.0, -300.0, top + 50.0])        # a high point printed earlier
    res = SimpleNamespace(q=np.array([_state(pl, pm, DOWN), q0, q1]),
                          toolpath=SimpleNamespace(dirs=np.array([DOWN] * 3)), segments=[(0, 0), (1, 1), (2, 2)])
    rows = [(pm, None), (p0, None), (p1, None)]
    t01, t12 = plan_transits(tpl, res, rows)
    assert t12.z_printed_mm == pytest.approx(pm[2])
    assert t12.z_mm >= pm[2] + cfg.planner["transit"]["printed_clearance_mm"] - 1e-9


def test_wrist_turn_over_180_is_flagged(cell):
    """P4-4: |dA4| or |dA6| > 180 deg in a transit is flagged (hose / cable wrap not modelled)."""
    _, robot, _, tpl = cell
    q0 = np.radians([0, -90, 90, 0, 45, 0])
    for j, turn, flagged in ((5, 200, True), (3, -190, True), (5, 170, False)):
        q1 = q0.copy()
        q1[j] += np.radians(turn)
        m = Move("PTP", "transit", np.array([q0, q1]), robot.fk(np.array([q0, q1]))[:, :3, 3], True)
        f = tpl.flags(Transit(0, 1, 0.0, moves=[m]))
        assert any(f"A{j + 1} turns" in x and "wrap" in x for x in f) == flagged, (j, turn, f)


def test_exported_perimeter_transits(cell, fresh_plans, tmp_path):
    """P4-9: export a fresh perimeter plan into tmp_path (not the committed .src). Every move between
    segments starts and ends at or above the safe height (except the retract / approach LINs), each
    transit PTP has its own $VEL_AXIS, and the header has the flags and the clearance per move."""
    cfg, robot, pl, tpl = cell
    _, _, _, plans = fresh_plans
    res = plans["perimeter"]
    f = tmp_path / "perimeter.src"
    rep = write_krl(f, robot, res, outputs.tcp_rows(robot, res), cfg, "perimeter", tpl)
    assert b"\r" not in f.read_bytes()                                        # K-2: LF only
    safe = cfg.bed()["top_z"] + cfg.planner["transit"]["safe_height_mm"]
    lines = f.read_text().splitlines()
    assert lines[0] == NOT_RUN
    assert any(ln.startswith("; FLAGS:") for ln in lines)
    for i in range(1, 7):
        assert f"  $VEL_AXIS[{i}] = {cfg.planner['transit']['vel_axis_pct']:g}" in lines
        assert f"  $ACC_AXIS[{i}] = {cfg.planner['transit']['acc_axis_pct']:g}" in lines
    # P4-5: min clearance per move in the header and in the report
    for t in rep["transits"]:
        per = t["move_min_clearance_mm"]
        assert {"retract", "transit", "approach"} <= set(per)
        assert min(per.values()) == pytest.approx(t["min_clearance_mm"])
        head = next(ln for ln in lines if ln.startswith(f"; transit {t['from_segment']}->{t['to_segment']}:"))
        for k, v in per.items():
            assert f"{k} {v:.1f}" in head
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
    assert n_transit == len(res.segments) - 1 == len(rep["transits"]) == 2
    ptps = _ptp_blocks(lines)
    assert [v for v, _ in ptps] == [[t["ptp_vel_axis_pct"]] * 6 for t in rep["transits"]]
    lim = cfg.planner["transit"]["speed_safety_factor"] * cfg.planner["transit"]["max_tcp_speed_mm_s"]
    assert all(t["peak_tcp_mm_s_model"] <= lim for t in rep["transits"])


def test_committed_src_files_are_lf():
    """K-2: every committed .src has LF line endings (see also .gitattributes)."""
    files = sorted((REPO / "results").glob("*/*.src"))
    assert files
    for f in files:
        assert b"\r" not in f.read_bytes(), f

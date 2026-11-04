import re

import numpy as np
import pytest

from kukasim import hilbert_corner as hc
from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim.krl import NOT_RUN

# A hand-placed square inside the reachable bed: skips the reachability fit (faster).
SQUARE = {"side_mm": 300.0, "center_xy": (690.0, -400.0)}


@pytest.fixture(scope="module")
def cell():
    cfg = load_config()
    return cfg, Robot.from_config(cfg)


@pytest.fixture(scope="module")
def default(cell):
    cfg, robot = cell
    return hc.generate(cfg, robot)


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_hilbert_geometry(order):
    side, c = 210.0, (700.0, -400.0)
    P = hc.hilbert_path(order, side, c, 2.25)
    n = 1 << order
    step = side / (n - 1)
    assert len(P) == n * n
    # every grid point exactly once
    g = np.round((P[:, :2] - P[:, :2].min(0)) / step).astype(int)
    assert len({tuple(v) for v in g}) == n * n and g.max() == n - 1
    # continuity: every segment is one grid step along X or Y
    d = np.diff(P[:, :2], axis=0)
    assert np.allclose(np.linalg.norm(d, axis=1), step)
    assert np.allclose(np.abs(d).min(1), 0)
    # square of the given side around the centre, constant Z
    assert np.allclose(P[:, :2].max(0) - P[:, :2].min(0), side)
    assert np.allclose(0.5 * (P[:, :2].max(0) + P[:, :2].min(0)), c)
    assert np.allclose(P[:, 2], 2.25)
    # open curve: start and end are the two adjacent corners of one side, not the same point
    assert np.isclose(np.linalg.norm(P[0, :2] - P[-1, :2]), side)


def test_hilbert_corner_count():
    # order 1 is a U (2 corners); order 2 counted by hand: 12 corners, 3 straight-through vertices
    for order, expect in ((1, 2), (2, 12)):
        idx, turn = hc.find_corners(hc.hilbert_path(order, 100.0, (0, 0), 0))
        assert len(idx) == expect
        assert set(np.abs(turn)) == {1}
    # every corner is 90 deg: neighbouring segments are perpendicular
    P = hc.hilbert_path(3, 100.0, (0, 0), 0)
    idx, _ = hc.find_corners(P)
    d = np.diff(P[:, :2], axis=0)
    assert np.allclose(np.einsum("ij,ij->i", d[idx - 1], d[idx]), 0)
    # non-corner interior vertices are straight-through
    rest = np.setdiff1d(np.arange(1, len(P) - 1), idx)
    assert np.allclose(d[rest - 1, 0] * d[rest, 1] - d[rest - 1, 1] * d[rest, 0], 0)


def test_default_layout(default):
    ct = default
    assert ct.params["order"] == 3
    assert ct.reach is not None and ct.side >= 200            # fitted to the reachable bed
    assert ct.kinds[0] == "start" and ct.kinds[-1] == "end"
    assert sum(k == "corner" for k in ct.kinds) == len(hc.find_corners(ct.points)[0])
    assert np.isclose(ct.seg.sum(), ct.s_w[-1])
    assert np.isclose(ct.seg.sum(), ct.side / 7 * 63)            # 63 grid steps of side / 7
    assert ct.ok, [c for c in ct.checks if not c["ok"]]


# ------------------------------------------------------------------ schedule
def test_schedule_coverage_and_seed(default, cell):
    ct = default
    is_c = np.array([k == "corner" for k in ct.kinds])
    used = ct.r_cmd[is_c & ~ct.excluded]
    for r in ct.params["ratios"]:
        assert np.isclose(used, r).sum() >= ct.params["min_repeats"], r
    assert used.min() < 1 < used.max()
    # seeded reproducibility of the assignment
    elig = np.ones(40, bool)
    elig[[3, 17]] = False
    a = hc.assign_ratios(40, elig, ct.params["ratios"], 2, seed=7)
    b = hc.assign_ratios(40, elig, ct.params["ratios"], 2, seed=7)
    c = hc.assign_ratios(40, elig, ct.params["ratios"], 2, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a[3] == 1 and a[17] == 1
    for r in ct.params["ratios"]:
        assert np.isclose(a[elig], r).sum() >= 38 // 7         # spread evenly, remainder on distinct ratios
    cfg, robot = cell
    again = hc.generate(cfg, robot, seed=ct.params["seed"], side_mm=ct.side, center_xy=ct.center)
    assert np.array_equal(again.r_cmd, ct.r_cmd)


def test_override_quantised_and_below_100(cell):
    """pc mode: integer $OV_PRO <= 100 %."""
    cfg, robot = cell
    ct = hc.generate(cfg, robot, speed_mode="pc", **SQUARE)
    is_c = np.array([k == "corner" for k in ct.kinds])
    ov = ct.ov_corner[is_c]
    assert np.allclose(ov, np.round(ov)) and ov.max() <= 100
    assert np.allclose(ct.r_eff[is_c], ct.r_cmd[is_c], atol=0.01)


def test_krl_mode_sets_the_corner_speed_in_the_program(default, cell, tmp_path):
    """P3-2: default mode: each corner leg is split at +-d_zone and the corner sub-segments get
    $VEL.CP = r x v, so the ratio is path-synchronous in the program itself."""
    cfg, robot = cell
    ct = default
    p = ct.params
    assert ct.mode == "krl" and p["d_zone_mm"] >= 2 * p["c_dis_mm"]
    v = p["straight_speed_mm_s"]
    f = tmp_path / "k.src"
    hc.write_krl(f, ct, robot, cfg)
    lines = f.read_text().splitlines()
    assert not any("PC-side" in ln for ln in lines)
    body = lines[lines.index("DEF hilbert_corner()"):]
    vel = None
    pts = []          # (xyz, $VEL.CP in mm/s for the motion INTO this point)
    for ln in body:
        m = re.match(r"  \$VEL\.CP = ([\d.]+)", ln)
        if m:
            vel = float(m.group(1)) * 1000.0
        if ln.startswith("  LIN"):
            pts.append((np.array([float(x) for x in re.findall(r"[XYZ] (-?[\d.]+)", ln)]), vel, ln))
    is_c = np.array([k == "corner" for k in ct.kinds])
    for j in np.nonzero(is_c)[0]:
        k = next(i for i, (x, _, _) in enumerate(pts) if np.allclose(x, ct.W[j], atol=1e-3))
        (a, va, la), (c, vc, lc), (b, vb, lb) = pts[k - 1], pts[k], pts[k + 1]
        assert "zone start" in la and "zone end" in lb and all("C_DIS" in x for x in (la, lc, lb))
        assert np.linalg.norm(c - a) == pytest.approx(p["d_zone_mm"], abs=2e-3)
        assert np.linalg.norm(b - c) == pytest.approx(p["d_zone_mm"], abs=2e-3)
        assert va == pytest.approx(v)                                  # straight up to the zone start
        assert vc == pytest.approx(v * ct.r_cmd[j], abs=0.006)        # both corner sub-segments at r x v
        assert vb == pytest.approx(v * ct.r_cmd[j], abs=0.006)
        assert pts[k + 2][1] == pytest.approx(v)                       # back to v after the zone
    # the model uses the same zones, with no PC latency
    assert ct.params["latency_s"] == 0.0
    assert np.allclose(ct.windows[is_c, 1] - ct.windows[is_c, 0], 2 * p["d_zone_mm"])
    # d_zone < 2 x C_DIS is rejected
    with pytest.raises(ValueError, match="d_zone_mm .* must be >= 2 x c_dis_mm"):
        hc.generate(cfg, robot, d_zone_mm=5.0, c_dis_mm=3.0, **SQUARE)


def test_pc_mode_is_an_option_and_says_it_needs_a_follower(cell, tmp_path):
    cfg, robot = cell
    ct = hc.generate(cfg, robot, speed_mode="pc", **SQUARE)
    hc.write_krl(tmp_path / "pc.src", ct, robot, cfg)
    text = (tmp_path / "pc.src").read_text()
    assert "PC-side schedule follower (NOT provided in this" in text
    assert sum(ln.startswith("  $VEL.CP") for ln in text.splitlines()) == 1 and "zone start" not in text
    hc.write_report(tmp_path / "r.md", ct, cfg, [])
    assert "does NOT provide" in (tmp_path / "r.md").read_text()
    with pytest.raises(ValueError, match="speed_mode"):
        hc.generate(cfg, robot, speed_mode="ov", **SQUARE)


def test_min_straight_rule(cell):
    # synthetic: corner 2 has a short straight after it, so corners 2 and 3 are too close
    kinds = ["start", "corner", "corner", "corner", "end"]
    seg = np.array([50.0, 50.0, 25.0, 50.0])
    p = {"window_before_mm": 10, "window_after_mm": 5}
    s_in, s_out = hc.corner_windows(seg, kinds, p)
    assert list(hc.too_close(s_in, s_out, kinds, 15.0)) == [False, False, True, True, False]
    assert not hc.too_close(s_in, s_out, kinds, 10.0).any()
    # full path, side 300 mm: grid step 42.9 mm, one-step straight between windows 27.9 mm
    cfg, robot = cell
    ok = hc.generate(cfg, robot, min_straight_mm=20.0, speed_mode="pc", **SQUARE)
    assert not ok.excluded.any()
    ct = hc.generate(cfg, robot, min_straight_mm=30.0, speed_mode="pc", **SQUARE)
    is_c = np.array([k == "corner" for k in ct.kinds])
    short = np.fmin(ct.straight_in, ct.straight_out) < 30.0
    assert (ct.excluded == (is_c & short)).all() and ct.excluded.any()
    assert np.all(ct.r_cmd[ct.excluded] == 1.0)
    chk = next(c for c in ct.checks if c["name"].startswith("min straight"))
    assert not chk["ok"]
    listed = {int(v) for v in chk["detail"].split(":")[1].split(",")}
    assert listed == set(ct.corner_ids[ct.excluded].tolist())
    # with every corner excluded the schedule cannot cover the ratios: flagged as an error
    assert not ct.ok
    # straight length = vertex distance minus the override windows at both ends
    j = int(np.nonzero(is_c)[0][3])
    assert np.isclose(ct.straight_in[j], ct.seg[j - 1] - 10.0 - (5.0 if ct.kinds[j - 1] == "corner" else 0.0))


def test_corner_windows_start_end():
    p = {"window_before_mm": 10, "window_after_mm": 5}
    s_in, s_out = hc.corner_windows(np.array([30.0, 40.0, 20.0]), ["start", "corner", "corner", "end"], p)
    assert np.allclose(s_in[1:], [20.0, 25.0, 15.0]) and np.isnan(s_in[0])
    assert np.allclose(s_out[:-1], [20.0, 25.0, 15.0]) and np.isnan(s_out[-1])


# --------------------------------------------------------------- speed model
def test_speed_model(cell, default):
    ct = default
    is_c = np.array([k == "corner" for k in ct.kinds])
    # default: long straights, wide blend -> the model reaches the commanded ratio
    assert np.allclose(ct.ratio_pred[is_c], ct.r_eff[is_c], atol=0.02)
    assert np.allclose(ct.v_straight[1:], ct.params["straight_speed_mm_s"], rtol=1e-6)
    assert np.allclose(ct.r_blend[is_c], ct.d_eff[is_c] / np.sqrt(2), rtol=1e-3)
    assert (ct.d_eff[1:-1] <= 0.5 * np.minimum(ct.seg[:-1], ct.seg[1:]) + 1e-12).all()
    cfg, robot = cell
    # a tiny C_DIS caps the fast corners by the blend radius: predicted ratio below commanded
    small = hc.generate(cfg, robot, c_dis_mm=0.3, **SQUARE)
    fast = is_c & (small.r_eff > 1.2)
    vcap = np.sqrt(small.params["lateral_acc_mm_s2"] * 0.3 / np.sqrt(2))
    assert np.all(small.v_corner[fast] <= vcap + 1e-6)
    assert np.all(small.ratio_pred[fast] < small.r_eff[fast] - 0.05)
    assert not next(c for c in small.checks if c["name"].startswith("model"))["ok"]


# -------------------------------------------------------------------- checks
def _failed(ct):
    return {c["name"].split(":")[0] for c in ct.checks if not c["ok"] and c["severity"] == "error"}


def test_checks_flag_bad_configurations(cell, tmp_path):
    cfg, robot = cell
    # square on the unreachable +X/-Y corner, over the clamp and off the bed
    bad = hc.generate(cfg, robot, order=2, side_mm=200.0, center_xy=(980.0, -690.0))
    f = _failed(bad)
    assert not bad.ok
    assert any(n.startswith("reachable") for n in f)
    assert "nozzle over the bed outline" in f
    # too fast
    fast = hc.generate(cfg, robot, order=2, straight_speed_mm_s=80.0, straight_override_pct=60, **SQUARE)
    assert any(n.startswith("TCP speed") for n in _failed(fast))
    # override above 100 %
    ov = hc.generate(cfg, robot, order=2, straight_override_pct=80, speed_mode="pc", **SQUARE)
    assert "corner override inside 1..100 %" in _failed(ov)
    # layer too thin for the highest measured bed point
    thin_cfg = load_config(cell_overrides={"process": {"layer_height_mm": [0.3]}})
    thin = hc.generate(thin_cfg, robot, order=2, **SQUARE)
    assert any(n.startswith("min nozzle-to-bed") for n in _failed(thin))
    # too few corners for the ratios
    few = hc.generate(cfg, robot, order=1, **SQUARE)
    assert any(n.startswith("schedule") for n in _failed(few))
    # the KRL of a failed configuration still starts with NOT RUN and says it failed
    hc.write_krl(tmp_path / "bad.src", bad, robot, cfg)
    text = (tmp_path / "bad.src").read_text().splitlines()
    assert text[0] == NOT_RUN and any("CHECKS FAILED" in t for t in text)


def test_outputs(default, cell, tmp_path):
    cfg, robot = cell
    ct = default
    hc.write_krl(tmp_path / "a.src", ct, robot, cfg)
    lines = (tmp_path / "a.src").read_text().splitlines()
    assert lines[0] == NOT_RUN
    lin = [ln for ln in lines if ln.strip().startswith("LIN")]
    n_c = sum(k == "corner" for k in ct.kinds)
    # waypoints + lift at the end + the two zone ends per corner (krl mode), no densification
    assert len(lin) == len(ct.W) + 1 + 2 * n_c
    assert sum("C_DIS" in ln for ln in lin) == 3 * n_c
    assert b"\r" not in (tmp_path / "a.src").read_bytes()               # K-2: LF only
    # P3-6: every LIN longer than min_lin_mm
    xyz = np.array([[float(x) for x in re.findall(r"[XYZ] (-?[\d.]+)", ln)] for ln in lin])
    assert np.linalg.norm(np.diff(xyz, axis=0), axis=1).min() > ct.params["min_lin_mm"]
    hc.write_schedule_csv(tmp_path / "s.csv", ct)
    rows = (tmp_path / "s.csv").read_text().splitlines()
    assert len(rows) == 1 + sum(k == "corner" for k in ct.kinds)
    hc.write_waypoints_csv(tmp_path / "w.csv", ct, robot)
    assert len((tmp_path / "w.csv").read_text().splitlines()) == 1 + len(ct.W)


# ----------------------------------------------------- review follow-ups (#3)
def test_header_and_scope_of_the_checks(default, cell, tmp_path):
    """P3-1 / K-1: 'checked against cell.yaml only', controller defaults stated, $APO.CDIS explicit."""
    cfg, robot = cell
    hc.write_krl(tmp_path / "a.src", default, robot, cfg)
    lines = (tmp_path / "a.src").read_text().splitlines()
    chk = next(ln for ln in lines if ln.startswith("; Checks:"))
    assert "all passed" not in chk
    assert "checked against cell.yaml only; site keep-outs must be checked locally before running" in chk
    assert any("INI / BAS(#INITMOV,0)" in ln and "$ACC.CP" in ln and "$VEL.ORI" in ln for ln in lines)
    assert f"  $APO.CDIS = {default.params['c_dis_mm']:g}" in lines


def test_default_square_keeps_a_margin_from_the_clamps(default, cell):
    """P3-1: the default square is smaller and stays clamp_margin_mm (XY) away from the clamp boxes."""
    cfg, robot = cell
    ct = default
    assert ct.side <= ct.params["max_side_mm"] and ct.side < 380
    from kukasim.collision import CollisionModel
    boxes = CollisionModel(robot, cfg).boxes[1:]
    d = hc.box_xy_distance(ct.dense["P"][:, 0], ct.dense["P"][:, 1], boxes).min()
    assert d >= ct.params["clamp_margin_mm"]
    chk = next(c for c in ct.checks if c["name"].startswith("XY margin"))
    assert chk["ok"]
    # a hand-placed square next to the +X/+Y clamp fails that check
    near = hc.generate(cfg, robot, order=2, side_mm=150.0, center_xy=(840.0, -260.0))
    assert any(n.startswith("XY margin") for n in _failed(near))


def test_keep_out_list(cell, default):
    """P3-1: optional cell.yaml keep_out boxes (empty in the repository) are checked like the clamps."""
    cfg, robot = cell
    assert cfg.cell["keep_out"] == []
    chk = next(c for c in default.checks if c["name"].startswith("keep-out"))
    assert chk["ok"] and "0 zone(s)" in chk["value"]
    cx, cy = np.asarray(SQUARE["center_xy"]) - SQUARE["side_mm"] / 2          # the start of the path
    box = {"name": "test_box", "x": [cx - 20, cx + 20], "y": [cy - 20, cy + 20], "z": [0, 200]}
    ko_cfg = load_config(cell_overrides={"keep_out": [box]})
    ct = hc.generate(ko_cfg, robot, order=2, **SQUARE)
    f = _failed(ct)
    assert any(n.startswith("keep-out zones") for n in f) and not ct.ok
    with pytest.raises(ValueError, match="keep_out 'broken'"):
        hc.generate(load_config(cell_overrides={"keep_out": [{"name": "broken", "x": [0, 1]}]}), robot, **SQUARE)


def test_layer_z_law(cell):
    """P3-3: z(x, y) = z0 + kx (x - x0) + ky (y - y0); default flat at top_z + layer height."""
    cfg, robot = cell
    flat = hc.generate(cfg, robot, order=2, **SQUARE)
    z0 = cfg.bed()["top_z"] + cfg.layer_height
    assert np.allclose(flat.W[:, 2], z0) and np.allclose(flat.dense["P"][:, 2], z0)
    law = {"z0": 5.0, "kx": 0.005, "ky": -0.0025, "x0": 700.0, "y0": -400.0}
    t = hc.generate(cfg, robot, layer_z=law, **SQUARE)
    W = t.W
    assert np.allclose(W[:, 2], 5.0 + 0.005 * (W[:, 0] - 700.0) - 0.0025 * (W[:, 1] + 400.0))
    assert np.ptp(W[:, 2]) > 1.0
    P = t.dense["P"]
    assert np.allclose(P[:, 2], 5.0 + 0.005 * (P[:, 0] - 700.0) - 0.0025 * (P[:, 1] + 400.0), atol=1e-6)
    assert t.ok, [c for c in t.checks if not c["ok"]]
    # x0, y0 default to the square centre
    c = hc.generate(cfg, robot, order=2, layer_z={"kx": 0.01}, **SQUARE)
    assert np.allclose(c.W[:, 2], z0 + 0.01 * (c.W[:, 0] - SQUARE["center_xy"][0]))


def test_approach_ptp(default, cell, tmp_path):
    """P3-4 / P3-7: the approach PTP target comes from IK with the hard limits, is collision-free,
    and gets $VEL_AXIS / $ACC_AXIS before it."""
    cfg, robot = cell
    ct = default
    from kukasim.planner import Planner
    pl = Planner(robot, cfg)
    q = ct.approach_q
    assert q is not None
    assert ((q >= pl.hard_limits[:, 0]) & (q <= pl.hard_limits[:, 1])).all()
    assert pl.coll.clearance(q)[0] >= 0
    assert np.allclose(robot.fk(q)[:3, 3], ct.W[0] + [0, 0, ct.params["approach_lift_mm"]], atol=1e-6)
    hc.write_krl(tmp_path / "a.src", ct, robot, cfg)
    lines = (tmp_path / "a.src").read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("  PTP"))
    before = lines[:k]
    for i in range(1, 7):
        assert f"  $VEL_AXIS[{i}] = {ct.params['approach_vel_axis_pct']:g}" in before
        assert f"  $ACC_AXIS[{i}] = {ct.params['approach_acc_axis_pct']:g}" in before


def test_corner_dip_factor(cell, default, tmp_path):
    """P3-5: default 1.0 keeps the old model; a dip factor lowers the predicted corner ratio;
    the width column is labelled model-only and the report says the logged speed decides."""
    cfg, robot = cell
    is_c = np.array([k == "corner" for k in default.kinds])
    assert default.params["corner_dip_factor"] == 1.0
    assert np.allclose(default.ratio_pred[is_c], default.r_eff[is_c], atol=0.02)
    dip = hc.generate(cfg, robot, corner_dip_factor=0.8, **SQUARE)
    assert np.allclose(dip.ratio_pred[is_c], 0.8 * dip.r_eff[is_c], atol=0.02)
    assert not next(c for c in dip.checks if c["name"].startswith("model"))["ok"]
    hc.write_waypoints_csv(tmp_path / "w.csv", default, robot)
    hdr = (tmp_path / "w.csv").read_text().splitlines()[0].split(",")
    assert "model_only_width_rel" in hdr and "model_width_rel" not in hdr
    hc.write_report(tmp_path / "r.md", default, cfg, [])
    rep = (tmp_path / "r.md").read_text()
    assert "The logged speed decides" in rep and "model only" in rep


@pytest.mark.parametrize("kw, msg", [
    ({"side_mm": 40.0}, "corner zones overlap"),                     # the case that divided by zero
    ({"side_mm": 40.0, "speed_mode": "pc"}, "corner zones overlap"),
    ({"side_mm": 14.0, "c_dis_mm": 1.5, "d_zone_mm": 3.0}, "shorter than 2 x C_DIS"),
    ({"side_mm": 7 * 12.4, "d_zone_mm": 6.0}, "LIN would be 0.40 mm"),
    ({"side_mm": 100.0, "speed_mode": "pc"}, "corner zones overlap"),
])
def test_geometry_guards_fail_loudly(cell, kw, msg):
    """P3-6: segment >= 2 x C_DIS, corner zones disjoint, every LIN > 0.5 mm, no division by zero."""
    cfg, robot = cell
    with pytest.raises(ValueError, match=msg):
        hc.generate(cfg, robot, center_xy=SQUARE["center_xy"], **kw)


def test_committed_src_is_lf():
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "results" / "hilbert_corner" / "hilbert_corner.src"
    assert b"\r" not in f.read_bytes()

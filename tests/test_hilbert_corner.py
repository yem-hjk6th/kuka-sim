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


def test_override_quantised_and_below_100(default):
    ct = default
    is_c = np.array([k == "corner" for k in ct.kinds])
    ov = ct.ov_corner[is_c]
    assert np.allclose(ov, np.round(ov)) and ov.max() <= 100
    assert np.allclose(ct.r_eff[is_c], ct.r_cmd[is_c], atol=0.01)


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
    ok = hc.generate(cfg, robot, min_straight_mm=20.0, **SQUARE)
    assert not ok.excluded.any()
    ct = hc.generate(cfg, robot, min_straight_mm=30.0, **SQUARE)
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
    ov = hc.generate(cfg, robot, order=2, straight_override_pct=80, **SQUARE)
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
    assert len(lin) == len(ct.W) + 1                       # waypoints + lift at the end, no densification
    assert sum("C_DIS" in ln for ln in lin) == sum(k == "corner" for k in ct.kinds)
    hc.write_schedule_csv(tmp_path / "s.csv", ct)
    rows = (tmp_path / "s.csv").read_text().splitlines()
    assert len(rows) == 1 + sum(k == "corner" for k in ct.kinds)
    hc.write_waypoints_csv(tmp_path / "w.csv", ct, robot)
    assert len((tmp_path / "w.csv").read_text().splitlines()) == 1 + len(ct.W)

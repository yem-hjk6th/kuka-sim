from pathlib import Path

import numpy as np
import pytest

from kukasim import outputs
from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim.krl import NOT_RUN, write_krl
from kukasim.planner import Planner, States, tcp_poses
from kukasim.rotations import rotation_angle_between
from kukasim.toolpaths import SURFACES, Toolpath, make_toolpath


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(planner_overrides={"toolpath": {"patch_half_size": 40, "raster_spacing": 40}})
    robot = Robot.from_config(cfg)
    return cfg, robot, Planner(robot, cfg)


def _sub(tp, a, b):
    return Toolpath(tp.name, tp.points[a:b], tp.dirs[a:b], tp.tilt_deg[a:b], tp.normal_tilt_deg[a:b],
                    tp.s[a:b] - tp.s[a], tp.max_tilt_deg)


def test_dp_equals_brute_force(setup):
    cfg, robot, pl = setup
    rng = np.random.default_rng(3)
    tp = make_toolpath(cfg, "dome", 30)
    for trial in range(6):
        i0 = int(rng.integers(0, len(tp.points) - 5))
        sub = _sub(tp, i0, i0 + 4)
        states, _ = pl.build_states(sub)
        # keep a random handful of states per point so brute force stays small
        small = []
        for S in states:
            k = rng.choice(len(S.node), size=min(7, len(S.node)), replace=False)
            small.append(States(S.bins[k], S.slots[k], S.q[k], S.node[k], S.limit_margin[k], S.clearance[k]))
        dq_max = pl.step_limits(sub.points)
        if trial % 2:
            dq_max = dq_max * 0.05          # make some transitions infeasible
        choice, segments, breaks, no_state, total = pl.dp(small, dq_max)
        best, seq = pl.brute_force(small, dq_max)
        if np.isfinite(best):
            assert segments == [(0, 3)]
            assert total == pytest.approx(best, rel=1e-9, abs=1e-12)
        else:
            assert len(segments) > 1 or breaks


def test_dp_reports_infeasible_points(setup):
    cfg, robot, pl = setup
    tp = make_toolpath(cfg, "sine_wave", 0)
    sub = _sub(tp, 0, 30)
    far = sub.points.copy()
    far[10:15] += np.array([2000.0, 0, 0])            # out of reach
    sub = Toolpath("t", far, sub.dirs, sub.tilt_deg, sub.normal_tilt_deg, sub.s, 0)
    res = pl.plan(sub)
    assert res.no_state == [(10, 14)]
    assert res.segments[0][0] == 0 and res.segments[-1][1] == 29
    assert np.isnan(res.q[10:15]).all() and not res.complete


@pytest.fixture(scope="module")
def planned(setup):
    cfg, robot, pl = setup
    out = {}
    for name in SURFACES:
        tp = make_toolpath(cfg, name, 30)
        if name == "perimeter":
            tp = _sub(tp, 200, 320)
        out[name] = pl.plan(tp)
    return out


def test_constraints_on_every_output_point(setup, planned):
    cfg, robot, pl = setup
    for name, res in planned.items():
        f = res.feasible
        assert f.any(), name
        q = res.q[f]
        lim = pl.hard_limits
        assert ((q >= lim[:, 0] - 1e-9) & (q <= lim[:, 1] + 1e-9)).all(), name
        # pose: tool X along the nozzle direction, TCP on the path
        T = robot.fk(q)
        assert np.abs(T[:, :3, 3] - res.toolpath.points[f]).max() < 1e-6
        assert np.abs(np.einsum("ij,ij->i", T[:, :3, 0], res.toolpath.dirs[f]) - 1).max() < 1e-9
        # collision-free
        assert (pl.coll.clearance(q) >= 0).all(), name
        # joint steps inside each continuous segment
        for a, b in res.segments:
            dq = np.abs(np.diff(res.q[a:b + 1], axis=0))
            assert (dq <= res.dq_max[a + 1:b + 1] + 1e-12).all(), name
            db = pl.bin_dist(res.bins[a:b][:, None], res.bins[a + 1:b + 1][:, None])
            assert (db <= pl.p["max_dbin"]).all()


def test_redundancy_angle_is_rotation_about_nozzle(setup):
    cfg, robot, pl = setup
    p = np.array([[713.0, -410.0, 20.0]])
    d = np.array([[0.3, -0.2, -0.93]])
    d /= np.linalg.norm(d)
    T = tcp_poses(p, d, pl.thetas)[0]
    assert np.allclose(T[:, :3, 0], d)                       # tool X fixed
    ang = np.degrees(rotation_angle_between(T[0, :3, :3], T[:, :3, :3]))
    expect = np.abs((np.degrees(pl.thetas - pl.thetas[0]) + 180) % 360 - 180)
    assert np.allclose(ang, expect, atol=1e-6)
    # theta = 0 with a vertical nozzle is the planar programs' A0 B90 C0
    T0 = tcp_poses(p, np.array([[0, 0, -1.0]]), np.array([0.0]))[0, 0]
    from kukasim.rotations import abc_to_matrix
    assert np.allclose(T0[:3, :3], abc_to_matrix(0, 90, 0), atol=1e-12)


def test_krl_not_run_header(setup, planned, tmp_path):
    cfg, robot, pl = setup
    for name, res in planned.items():
        rows = outputs.tcp_rows(robot, res)
        f = tmp_path / f"{name}.src"
        write_krl(f, robot, res, rows, cfg, name)
        assert f.read_text().splitlines()[0] == NOT_RUN


def test_not_run_header_in_every_committed_krl():
    repo = Path(__file__).resolve().parent.parent
    for f in list(repo.rglob("*.src")):
        if ".git" in f.parts:
            continue
        assert f.read_text().splitlines()[0] == NOT_RUN, f


def test_no_robot_communication_code():
    """Nothing in the package opens a network connection or talks to a controller."""
    repo = Path(__file__).resolve().parent.parent
    banned = ("import socket", "from socket", "openshowvar", "kukavarproxy", "import serial",
              "urllib", "requests", "http.client")
    files = list((repo / "kukasim").glob("*.py")) + list((repo / "scripts").glob("*.py")) + \
        [repo / "validate_fk.py"]
    for f in files:
        text = f.read_text().lower()
        for b in banned:
            assert b not in text, (f.name, b)

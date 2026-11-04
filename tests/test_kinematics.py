import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kukasim.config import load_config
from kukasim.kinematics import Robot, T_to_xyzabc, xyzabc_to_T
from kukasim.rotations import abc_to_matrix, matrix_to_abc, rotation_angle_between


@pytest.fixture(scope="module")
def robot():
    return Robot.from_config(load_config())


def test_abc_matches_scipy_intrinsic_zyx():
    rng = np.random.default_rng(1)
    for a, b, c in rng.uniform([-180, -89, -180], [180, 89, 180], size=(200, 3)):
        R = abc_to_matrix(a, b, c)
        Rs = Rotation.from_euler("ZYX", [a, b, c], degrees=True).as_matrix()
        assert np.allclose(R, Rs, atol=1e-12)
        assert np.allclose(matrix_to_abc(R), (a, b, c), atol=1e-8)


def test_abc_near_b90_singularity_is_rotation_exact():
    # the logged print pose: only A - C is defined at B = 90
    for abc in [(84.1, 89.9, 116.2), (84.1, 90.0, 116.2), (10, -90, 30), (0, 90, 0)]:
        R = abc_to_matrix(*abc)
        back = abc_to_matrix(*matrix_to_abc(R, c_hint_deg=abc[2]))
        assert np.degrees(rotation_angle_between(R, back)) < 1e-6
    a, b, c = matrix_to_abc(abc_to_matrix(84.1, 90.0, 116.2), c_hint_deg=116.2)
    assert abs(c - 116.2) < 1e-9 and abs(a - 84.1) < 1e-6


def test_home_pose_flange(robot):
    T = robot.fk_flange(np.radians([0, -90, 90, 0, 0, 0]))
    assert np.allclose(T[:3, 3], [630, 0, 985], atol=1e-6)      # 25 + 515 + 90, 400 + 560 + 25
    a, b, c = matrix_to_abc(T[:3, :3])
    assert abs(b - 90) < 1e-6 and abs(a - c) < 1e-6              # KUKA FLANGE at home: A0 B90 C0


def test_fk_ik_round_trip_random(robot):
    rng = np.random.default_rng(0)
    lim = robot.limits
    q = rng.uniform(lim[:, 0], lim[:, 1], size=(3000, 6))
    T = robot.fk(q)
    Q, ok = robot.ik(T)
    assert ok.any(axis=1).all()
    # every returned solution reproduces the pose
    Tq = robot.fk(Q[ok])
    Tref = np.repeat(T[:, None], Q.shape[1], 1)[ok]
    assert np.abs(Tq[:, :3, 3] - Tref[:, :3, 3]).max() < 1e-6
    assert np.degrees(rotation_angle_between(Tq[:, :3, :3], Tref[:, :3, :3])).max() < 1e-7
    # the original joint vector is among them (away from the wrist singularity)
    away = np.abs(q[:, 4]) > np.radians(0.5)
    d = np.abs(Q - q[:, None]).max(-1)
    d[~ok] = np.inf
    assert (d.min(1)[away] < 1e-8).all()
    # every solution respects the limits
    assert ((Q[ok] >= lim[:, 0] - 1e-9) & (Q[ok] <= lim[:, 1] + 1e-9)).all()


def test_turn_ambiguity_enumerated(robot):
    q = np.radians([10, -70, 80, 178, 40, 200])
    Q, ok = robot.ik(robot.fk(q[None]))
    sols = np.degrees(Q[0][ok[0]])
    same_arm = sols[np.all(np.abs(sols[:, :3] - np.degrees(q[:3])) < 1e-6, axis=1)]
    a6 = sorted(set(np.round(same_arm[:, 5], 6)))
    a4 = sorted(set(np.round(same_arm[:, 3], 6)))
    assert any(abs(v - 200) < 1e-6 for v in a6) and any(abs(v + 160) < 1e-6 for v in a6)
    assert any(abs(v - 178) < 1e-6 for v in a4) and any(abs(v + 182) < 1e-6 for v in a4)


def test_ik_wrist_singularity(robot):
    q = np.radians([5, -80, 95, 30, 0.0, 40])
    T = robot.fk(q[None])
    Q, ok = robot.ik(T)
    assert ok.any()
    assert np.abs(robot.fk(Q[ok])[:, :3, 3] - T[0, :3, 3]).max() < 1e-6


def test_xyzabc_round_trip():
    p = [700, -400, 50, 84.1, 30, 116.2]
    assert np.allclose(T_to_xyzabc(xyzabc_to_T(p)), p, atol=1e-9)


def test_limits_from_config_match_urdf(robot):
    from kukasim.kinematics import parse_urdf
    cfg = load_config()
    urdf = np.degrees([[j.lower, j.upper] for j in parse_urdf(cfg.urdf_path)])
    assert np.allclose(urdf, cfg.joint_limits_deg, atol=0.01)


# ------------------------------------------------- near-singular IK grids
POS_TOL_MM = 1e-6      # = Robot.IK_TOL_POS_MM
ROT_TOL = 1e-9         # = Robot.IK_TOL_ROT (max rotation-matrix element difference)


def _perturbed(T, rng, n=4, scale=1e-12):
    """T (M,4,4) plus n copies with ~1e-12 position (mm) / orientation (rad)
    perturbations, so both roundings of a singular pose are exercised on every
    numpy build (a 1-ulp difference decided the old failure)."""
    from kukasim.rotations import axis_angle
    out = [T]
    for _ in range(n):
        P = T.copy()
        P[:, :3, 3] += rng.normal(scale=scale, size=(len(T), 3))
        ax = rng.normal(size=3)
        P[:, :3, :3] = axis_angle(ax, scale * rng.uniform(0.5, 2.0)) @ T[:, :3, :3]
        out.append(P)
    return np.concatenate(out)


def _assert_fk_ik_exact(robot, T, q_ref=None, arm_tol=1e-6):
    """Every slot marked ok reproduces T; every target has an ok slot; and if
    q_ref is given, one ok slot has its arm (A1-A3) within arm_tol rad."""
    Q, ok = robot.ik(T)
    assert ok.any(axis=1).all(), np.flatnonzero(~ok.any(axis=1))[:5]
    F = robot.fk(Q[ok])
    Tr = np.repeat(T[:, None], Q.shape[1], 1)[ok]
    dp = np.abs(F[:, :3, 3] - Tr[:, :3, 3]).max()
    dr = np.abs(F[:, :3, :3] - Tr[:, :3, :3]).max()
    assert dp <= POS_TOL_MM and dr <= ROT_TOL, (dp, dr)
    assert np.degrees(rotation_angle_between(F[:, :3, :3], Tr[:, :3, :3])).max() < 1e-6
    if q_ref is not None:
        d = np.abs(Q[..., :3] - q_ref[:, None, :3]).max(-1)
        d[~ok] = np.inf
        assert (d.min(1) < arm_tol).all(), d.min(1).max()


def _assert_original_among_ok(Q, ok, q_ref, rows, tol=1e-7):
    """For the selected rows, one slot marked ok equals q_ref on all six joints (within tol rad)."""
    d = np.abs(Q - q_ref[:, None]).max(-1)
    d[~ok] = np.inf
    best = d.min(1)[rows]
    assert rows.any() and (best < tol).all(), (np.flatnonzero(rows)[best >= tol][:5], best.max())


def test_original_among_ok_catches_a_dropped_wrist_twin(robot):
    """Follow-up of #6: an IK that dropped the slot holding the original solution (here: every
    slot of its wrist-flip half) still passes the FK-only check, but not the original-joint check."""
    q = np.radians([[5, -80, 95, 30, x5, 40] for x5 in (-3.0, -0.1, 0.1, 1.0, 3.0)])
    T = robot.fk(q)
    Q, ok = robot.ik(T)
    _assert_original_among_ok(Q, ok, q, np.ones(len(q), bool))
    d = np.abs(Q - q[:, None]).max(-1)
    hit = np.argmin(np.where(ok, d, np.inf), axis=1)
    w = (hit // 4) % 2                                   # slot index = (((s*2+e)*2+w)*2+t4)*2+t6
    drop = ((np.arange(Q.shape[1])[None] // 4) % 2) == w[:, None]
    ok_dropped = ok & ~drop
    F = robot.fk(Q[ok_dropped])                          # what the FK-only check sees: still exact
    Tr = np.repeat(T[:, None], Q.shape[1], 1)[ok_dropped]
    assert ok_dropped.any(1).all() and np.abs(F[:, :3, 3] - Tr[:, :3, 3]).max() <= POS_TOL_MM
    with pytest.raises(AssertionError):
        _assert_original_among_ok(Q, ok_dropped, q, np.ones(len(q), bool))


def test_ik_near_wrist_singularity_grid(robot):
    rng = np.random.default_rng(7)
    a5 = [-1e-6, -1e-9, -1e-12, 0.0, 1e-12, 1e-9, 1e-6, -3.0, -1.0, -0.1, 0.1, 1.0, 3.0]
    a4 = [-170.0, -90.0, -30.0, 0.0, 30.0, 90.0, 175.0]
    a6 = [-300.0, -40.0, 0.0, 40.0, 120.0, 200.0]
    arms = [(5, -80, 95), (-40, -60, 60), (120, -110, 130), (-150, -30, -20)]
    q = np.array([(*arm, x4, x5, x6) for arm in arms for x4 in a4 for x5 in a5 for x6 in a6], float)
    q = np.radians(q)
    T = robot.fk(q)
    Tp = _perturbed(T, rng)
    qp = np.tile(q, (len(Tp) // len(q), 1))
    _assert_fk_ik_exact(robot, Tp, qp)
    # The FK check above is the same one Robot.ik applies, so it cannot see a solution that IK
    # dropped (e.g. a lost wrist twin). Away from the singularity (|A5| >= 0.1 deg) the original
    # joint vector is unique, so it must itself be among the slots marked ok.
    Q, ok = robot.ik(Tp)
    _assert_original_among_ok(Q, ok, qp, np.abs(qp[:, 4]) >= np.radians(0.1))
    # A4 + A6 (the only defined combination at A5 = 0) is reproduced by the
    # solution with the original arm, i.e. no A4 / A6 split error
    Q, ok = robot.ik(T)
    exact = np.abs(q[:, 4]) == 0
    s = (np.abs(Q[..., :3] - q[:, None, :3]).max(-1) < 1e-6) & ok
    d = np.abs(np.angle(np.exp(1j * ((Q[..., 3] + Q[..., 5]) - (q[:, 3] + q[:, 5])[:, None]))))
    d[~s] = np.inf
    assert (d.min(1)[exact] < 1e-9).all()


def test_ik_near_elbow_singularity_grid(robot):
    # A3 at the stretched-arm angle (wrist centre, A3 and A2 in line) +- small offsets
    rng = np.random.default_rng(8)
    d3 = np.degrees(robot.delta3)
    q = np.radians([(a1, a2, d3 + e, a4, a5, a6)
                    for a1 in (0, 35, -100) for a2 in (-150, -100, -40, 20)
                    for e in (-1e-6, -1e-9, 0.0, 1e-9, 1e-6, 0.5, -0.5)
                    for a4 in (-60, 0, 100) for a5 in (-50, 0.0, 30) for a6 in (0, 75)])
    T = robot.fk(q)
    Tp = _perturbed(T, rng)
    # At the stretched arm A3 depends on the target like sqrt(distance to the
    # reach boundary): a 1e-12 perturbation of T moves the exact A2 / A3 by
    # ~1e-6 rad, so the arm is compared loosely; FK(ik(T)) == T is exact.
    _assert_fk_ik_exact(robot, Tp, np.tile(q, (len(Tp) // len(q), 1)), arm_tol=1e-5)


def _shoulder_singular_a2(robot, a3_deg):
    """A2 (deg) that puts the wrist centre on the A1 axis for the given A3."""
    from scipy.optimize import brentq

    def x1(a2):
        q = np.radians([0, a2, a3_deg, 0, 0, 0])
        return robot.wrist_center(robot.fk_flange(q))[0]
    grid = np.linspace(-189, 44, 467)
    vals = [x1(g) for g in grid]
    for g0, g1, v0, v1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if v0 * v1 < 0:
            return brentq(x1, g0, g1, xtol=1e-13)
    return None


def test_ik_near_shoulder_singularity_grid(robot):
    # wrist centre on (or within ~1e-6 mm .. 1 mm of) the A1 axis
    rng = np.random.default_rng(9)
    rows = []
    for a3 in (60.0, 100.0, 140.0):
        a2s = _shoulder_singular_a2(robot, a3)
        assert a2s is not None
        for da2 in (-1e-6, -1e-9, 0.0, 1e-9, 1e-6, 0.05):
            for a1 in (-120.0, 0.0, 45.0):
                for a4, a5, a6 in ((0, 40, 0), (90, -60, 30), (-150, 1e-7, 200), (20, 0.0, -20)):
                    rows.append((a1, a2s + da2, a3, a4, a5, a6))
    q = np.radians(rows)
    assert np.hypot(*robot.wrist_center(robot.fk_flange(q))[:, :2].T).min() < 1e-6
    T = robot.fk(q)
    # on the A1 axis A1 is arbitrary, so only FK(ik(T)) == T is asserted
    _assert_fk_ik_exact(robot, _perturbed(T, rng))

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

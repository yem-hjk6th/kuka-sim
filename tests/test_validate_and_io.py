import numpy as np

import validate_fk
from kukasim.config import load_config
from kukasim.kinematics import Robot, T_to_xyzabc
from kukasim.slicer_import import parse_src, toolpath_from_src


def test_validate_fk_zero_error_on_own_fk(tmp_path):
    cfg = load_config()
    robot = Robot.from_config(cfg)
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(10):
        q = rng.uniform(robot.limits[:, 0], robot.limits[:, 1]) * 0.6
        rows.append(list(np.degrees(q)) + T_to_xyzabc(robot.fk(q)))
    f = tmp_path / "pairs.csv"
    f.write_text("# test\nA1,A2,A3,A4,A5,A6,X,Y,Z,A,B,C\n" +
                 "\n".join(",".join(f"{v:.9f}" for v in r) for r in rows) + "\n")
    pairs = validate_fk.read_pairs(f)
    dp, pos, ori = validate_fk.validate(pairs, robot)
    assert pos.max() < 1e-5 and ori.max() < 1e-5
    assert validate_fk.main([str(f)]) == 0


def test_template_is_readable():
    from pathlib import Path
    tmpl = Path(validate_fk.__file__).parent / "data" / "fk_pairs_template.csv"
    assert validate_fk.read_pairs(tmpl).shape == (0, 12)
    assert validate_fk.main([str(tmpl)]) == 0


def test_slicer_src_reader(tmp_path):
    src = tmp_path / "a.src"
    src.write_text("DEF a()\nLIN {X 700, Y -400, Z 20, A 90, B 60, C 0} C_DIS\n"
                   "LIN {X 710.5, Y -400, Z 21, A 90, B 60, C 0} C_DIS\nEND\n")
    assert parse_src(src.read_text()).shape == (2, 6)
    tp = toolpath_from_src(src, ds=1.0)
    assert len(tp.points) >= 10
    assert np.allclose(tp.tilt_deg, 30, atol=1e-6)     # B = 90 - tilt

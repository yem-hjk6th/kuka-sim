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


def test_fixed_number_format():
    """Follow-up of #6: kukasim/krl.py and kukasim/outputs.py write numbers in one fixed form:
    never a negative zero, orientation A / C in (-180, 180] (-180 noise is written as 180),
    joint angles not wrapped."""
    import re
    from pathlib import Path

    from kukasim.krl import fang, fnum, lin_pose, ptp_axes
    assert fnum(-0.0, 4) == fnum(-4e-5, 4) == fnum(4e-5, 4) == "0.0000"
    assert fnum(-1e-13, 3) == "0.000" and fnum(-0.0006, 3) == "-0.001"
    for a in (-180.0, 180.0, -180.0 + 1e-9, 180.0 - 1e-9, -179.99996, 540.0, -540.0):
        assert fang(a, 4) == "180.0000", a
    assert fang(-179.9999, 4) == "-179.9999" and fang(360.0 - 1e-12, 4) == "0.0000"
    assert fang(190.0, 4) == "-170.0000"
    txt = lin_pose([1e-13, -0.0, -2e-9], (-180.0 + 1e-10, -1e-12, -180.0))
    assert txt == "{X 0.000, Y 0.000, Z 0.000, A 180.0000, B 0.0000, C 180.0000}"
    # the same orientation from both sides of +-180 gives the same text
    assert lin_pose([0, 0, 0], (180 - 1e-11, 90, -180 + 1e-11)) == lin_pose([0, 0, 0], (-180 + 1e-11, 90, 180 - 1e-11))
    assert ptp_axes([-1e-9, -90, 90, 185.0, -0.0, -350.0]) == \
        "{A1 0.0000, A2 -90.0000, A3 90.0000, A4 185.0000, A5 0.0000, A6 -350.0000}"
    # the committed results follow it
    repo = Path(__file__).resolve().parent.parent
    files = sorted(repo.glob("results/*/*.src")) + sorted(repo.glob("results/*/trajectory.csv")) \
        + sorted(repo.glob("results/*/waypoints.csv"))
    assert len(files) == 12
    for f in files:
        t = f.read_text()
        assert not re.search(r"(?<![\d.])-0\.0*(?![\d.]*[1-9])(?=[,}\s]|$)", t, re.M), f
        assert not re.search(r"\b[AC] -180\.0+\b", t), f
        if f.suffix == ".csv":                             # orientation columns (A6 may be -180 legitimately)
            import csv
            for r in csv.DictReader(t.splitlines()):
                assert r["A"] != "-180.0000" and r["C"] != "-180.0000", (f, r["i"] if "i" in r else r["wp"])

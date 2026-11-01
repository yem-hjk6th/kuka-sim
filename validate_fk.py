"""Compare the simulator's FK (URDF + TOOL 3 + BASE 0) with poses read on the pendant.

Input CSV columns: A1,A2,A3,A4,A5,A6,X,Y,Z,A,B,C  (deg / mm, KUKA $AXIS_ACT and $POS_ACT
with TOOL 3 and BASE 0 active). Lines starting with '#' are ignored.

    python validate_fk.py data/fk_pairs_template.csv
    python validate_fk.py my_pairs.csv --tool 225.55 -0.6 72.83 0 0 0

Orientation error is the geodesic angle between the two rotation matrices, so poses
near B = 90 (Euler singularity, e.g. A 84.1 B 89.9 C 116.2) are compared correctly
even when their A and C look very different. This script only reads a file.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from kukasim.config import load_config
from kukasim.kinematics import Robot, xyzabc_to_T
from kukasim.rotations import rotation_angle_between

COLUMNS = ["A1", "A2", "A3", "A4", "A5", "A6", "X", "Y", "Z", "A", "B", "C"]


def read_pairs(path):
    rows = []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for r in reader:
        rows.append([float(r[c]) for c in COLUMNS])
    return np.array(rows).reshape(-1, 12)


def validate(pairs, robot):
    q = np.radians(pairs[:, :6])
    T_fk = robot.fk(q)
    T_meas = np.stack([xyzabc_to_T(p) for p in pairs[:, 6:]]) if len(pairs) else np.zeros((0, 4, 4))
    dp = T_fk[:, :3, 3] - T_meas[:, :3, 3]
    pos = np.linalg.norm(dp, axis=1)
    ori = np.degrees(rotation_angle_between(T_fk[:, :3, :3], T_meas[:, :3, :3]))
    return dp, pos, ori


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--tool", nargs=6, type=float, help="override TOOL X Y Z A B C")
    args = ap.parse_args(argv)
    cfg = load_config()
    tool = args.tool or cfg.tool_xyzabc
    robot = Robot(cfg.urdf_path, tool, cfg.base_xyzabc, cfg.joint_limits_deg)
    pairs = read_pairs(args.csv)
    if len(pairs) == 0:
        print(f"{args.csv}: no pairs (template). Add rows read from the pendant.")
        return 0
    dp, pos, ori = validate(pairs, robot)
    print(f"tool = {tool}")
    print(f"{'#':>3} {'dX':>8} {'dY':>8} {'dZ':>8} {'|dp| mm':>8} {'ori deg':>8}")
    for i, (d, p, o) in enumerate(zip(dp, pos, ori)):
        print(f"{i:>3} {d[0]:8.3f} {d[1]:8.3f} {d[2]:8.3f} {p:8.3f} {o:8.4f}")
    print(f"position error: mean {pos.mean():.3f} mm, max {pos.max():.3f} mm, rms {np.sqrt((pos**2).mean()):.3f} mm")
    print(f"orientation error: mean {ori.mean():.4f} deg, max {ori.max():.4f} deg")
    print(f"mean offset (FK - pendant): {dp.mean(0).round(3).tolist()} mm "
          "(a constant offset points at TOOL or BASE, a pose-dependent one at the URDF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

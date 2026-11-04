"""Re-export the KRL files (and their transit reports) for the default-tilt toolpaths.

Simulation only. Replans each surface exactly as run_step1.py does (the planner is
deterministic), rewrites results/<surface>/<surface>.src and export_report.json,
and updates the "export" entries of results/summary.json. Plots, CSVs and the
tilt / location sweeps are left as they are.

    python scripts/export_krl.py --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kukasim import outputs  # noqa: E402
from kukasim.config import load_config  # noqa: E402
from kukasim.kinematics import Robot  # noqa: E402
from kukasim.krl import write_krl  # noqa: E402
from kukasim.planner import Planner  # noqa: E402
from kukasim.toolpaths import SURFACES, make_toolpath  # noqa: E402
from kukasim.transit import TransitPlanner  # noqa: E402

from run_step1 import export_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config()
    robot = Robot.from_config(cfg)
    planner = Planner(robot, cfg)
    tpl = TransitPlanner(robot, cfg, planner.coll, planner.hard_limits)
    tilt0 = cfg.planner["toolpath"]["max_tilt_deg"]
    sf = out / "summary.json"
    summary = json.loads(sf.read_text()) if sf.exists() else {"surfaces": {}}
    for name in SURFACES:
        tp = make_toolpath(cfg, name, tilt0)
        res = planner.plan(tp)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        csv_file = d / "trajectory.csv"
        if csv_file.exists():
            # the export must describe the same plan as the committed trajectory
            with open(csv_file) as f:
                old = [[float(r[f"A{j}"] or "nan") for j in range(1, 7)] for r in csv.DictReader(f)]
            if not np.allclose(np.nan_to_num(old), np.nan_to_num(np.degrees(res.q)), atol=1e-3):
                raise SystemExit(f"{name}: plan differs from {csv_file}; run run_step1.py instead")
        rows = outputs.tcp_rows(robot, res)
        rep = write_krl(d / f"{name}.src", robot, res, rows, cfg, name, tpl)
        (d / "export_report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
        summary["surfaces"].setdefault(name, {})["export"] = export_summary(rep)
        print(name, len(rep["transits"]), "transits, flags:", rep["flags"] or "none", flush=True)
    sf.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Re-export the KRL files (and their transit reports) for the default-tilt toolpaths.

Simulation only. Replans each surface exactly as run_step1.py does (the planner is
deterministic), rewrites results/<surface>/<surface>.src and export_report.json,
rewrites trajectory.csv (same plan: every column is checked against the committed
file to 1e-3, orientation A / C modulo 360, before it is replaced; only the number
format can change), and updates the
"export" entries of results/summary.json. Plots and the tilt / location sweeps are
left as they are.

    python scripts/export_krl.py --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kukasim import outputs  # noqa: E402
from kukasim.config import load_config  # noqa: E402
from kukasim.kinematics import Robot  # noqa: E402
from kukasim.krl import write_krl  # noqa: E402
from kukasim.planner import Planner  # noqa: E402
from kukasim.toolpaths import SURFACES, make_toolpath  # noqa: E402
from kukasim.transit import TransitPlanner  # noqa: E402

from run_step1 import export_summary  # noqa: E402


CSV_ATOL = 1e-3          # deg / mm / unitless; every column of trajectory.csv is written with >= 2 decimals
CSV_WRAPPED = ("A", "C")  # orientation angles written wrapped to (-180, 180]: compared modulo 360


def csv_differences(old_file, new_file, atol=CSV_ATOL):
    """Cells in which two trajectory.csv files differ by more than the number format: same header,
    same number of rows, and in every column either the same text or numbers within atol (the
    wrapped orientation columns A / C modulo 360). Returns a list of 'row r, column c: old -> new'."""
    with open(old_file) as f:
        old = list(csv.reader(f))
    with open(new_file) as f:
        new = list(csv.reader(f))
    if old[:1] != new[:1]:
        return [f"header {old[:1]} -> {new[:1]}"]
    if len(old) != len(new):
        return [f"{len(old) - 1} -> {len(new) - 1} rows"]
    head = old[0]
    bad = []
    for r, (a, b) in enumerate(zip(old[1:], new[1:])):
        if len(a) != len(b):
            bad.append(f"row {r}: {len(a)} -> {len(b)} cells")
            continue
        for c, x, y in zip(head, a, b):
            if x == y:
                continue
            try:
                d = abs(float(x) - float(y))
            except ValueError:
                bad.append(f"row {r}, column {c}: {x!r} -> {y!r}")
                continue
            if c in CSV_WRAPPED:
                d = abs((d + 180.0) % 360.0 - 180.0)
            if not d <= atol:
                bad.append(f"row {r}, column {c}: {x} -> {y}")
    return bad


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
        rows = outputs.tcp_rows(robot, res)
        fresh = csv_file.with_name(csv_file.name + ".new")
        outputs.write_csv(fresh, robot, res, rows)
        if csv_file.exists():
            # the export must describe the same plan as the committed trajectory: every column
            bad = csv_differences(csv_file, fresh)
            if bad:
                fresh.unlink()
                raise SystemExit(f"{name}: plan differs from {csv_file} ({'; '.join(bad[:5])}); "
                                 "run run_step1.py instead")
        fresh.replace(csv_file)
        rep = write_krl(d / f"{name}.src", robot, res, rows, cfg, name, tpl)
        (d / "export_report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
        summary["surfaces"].setdefault(name, {})["export"] = export_summary(rep)
        print(name, len(rep["transits"]), "transits, flags:", rep["flags"] or "none", flush=True)
    sf.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

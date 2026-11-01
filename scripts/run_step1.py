"""Step 1: plan every test toolpath, write per-path outputs and RESULTS.md.

Simulation only. Nothing here connects to a robot or a controller.

    python scripts/run_step1.py --out results            # full run (a few minutes)
    python scripts/run_step1.py --out out --quick         # smaller sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kukasim import outputs  # noqa: E402
from kukasim.config import load_config  # noqa: E402
from kukasim.kinematics import Robot  # noqa: E402
from kukasim.krl import write_krl  # noqa: E402
from kukasim.planner import Planner, diagnose, turn_window  # noqa: E402
from kukasim.toolpaths import SURFACES, make_toolpath  # noqa: E402


def summarize(res, diag):
    th = np.where(res.bins >= 0, res.thetas_deg[np.clip(res.bins, 0, None)], np.nan)
    tw = outputs.unwrap_deg(th)
    f = res.feasible
    return {
        "points": int(len(res.bins)),
        "length_mm": float(res.toolpath.s[-1]),
        "planned_pct": float(100 * f.mean()),
        "complete": bool(res.complete),
        "segments": len(res.segments),
        "no_state": [list(map(int, x)) for x in res.no_state],
        "breaks": list(map(int, res.breaks)),
        "theta_span_deg": float(np.nanmax(tw) - np.nanmin(tw)) if f.any() else None,
        "theta_range_deg": [float(np.nanmin(tw)), float(np.nanmax(tw))] if f.any() else None,
        "min_limit_margin_deg": float(np.nanmin(res.limit_margin)) if f.any() else None,
        "min_abs_a5_deg": float(np.nanmin(np.abs(np.degrees(res.q[:, 4])))) if f.any() else None,
        "min_clearance_mm": float(np.nanmin(res.clearance)) if f.any() else None,
        "max_tilt_used_deg": float(res.toolpath.tilt_deg.max()),
        "max_normal_tilt_deg": float(res.toolpath.normal_tilt_deg.max()),
        "cost": float(res.cost),
        "runtime_s": {k: round(v, 2) for k, v in res.runtime_s.items()},
        "stretches": diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    robot = Robot.from_config(cfg)
    planner = Planner(robot, cfg)
    tilt0 = cfg.planner["toolpath"]["max_tilt_deg"]
    sweep = cfg.planner["toolpath"]["tilt_sweep_deg"]
    if args.quick:
        sweep = [0, 30]
    summary = {"default_max_tilt_deg": tilt0, "surfaces": {}, "sweep": {}}

    for name in SURFACES:
        print(f"[{name}] default tilt {tilt0}", flush=True)
        tp = make_toolpath(cfg, name, tilt0)
        res = planner.plan(tp)
        diag = diagnose(planner, tp, res)
        d = out / name
        d.mkdir(exist_ok=True)
        rows = outputs.tcp_rows(robot, res)
        outputs.write_csv(d / "trajectory.csv", robot, res, rows)
        title = f"{name} (max tilt {tilt0:g})"
        outputs.plot_joints(d / "joints.png", robot, res, title)
        outputs.plot_margins(d / "margins.png", res, cfg, title)
        outputs.plot_feasibility(d / "feasibility.png", res, title)
        write_krl(d / f"{name}.src", robot, res, rows, cfg, name)
        if not args.no_anim:
            outputs.animate(d / "animation.gif", planner.coll, res, name)
        summary["surfaces"][name] = summarize(res, diag)
        print("   ", {k: summary["surfaces"][name][k] for k in
                      ("planned_pct", "segments", "no_state", "breaks", "runtime_s")}, flush=True)

    for name in SURFACES:
        if name == "perimeter":
            continue      # flat path: the tilt cap does not change it
        summary["sweep"][name] = {}
        for t in sweep:
            tp = make_toolpath(cfg, name, t)
            res = planner.plan(tp)
            s = summarize(res, diagnose(planner, tp, res))
            summary["sweep"][name][str(t)] = {k: s[k] for k in (
                "planned_pct", "complete", "segments", "no_state", "breaks", "theta_span_deg",
                "min_limit_margin_deg", "min_abs_a5_deg", "runtime_s", "max_tilt_used_deg")}
            summary["sweep"][name][str(t)]["reasons"] = sorted({x["reason"] for x in s["stretches"]})
            print(f"[sweep {name} tilt {t}] {s['planned_pct']:.1f}% segs {s['segments']} "
                  f"{s['runtime_s']['total_s']}s", flush=True)

    # Same surfaces moved to the four bed quadrants (location matters more than shape)
    bed = cfg.bed()
    a = cfg.planner["toolpath"]["patch_half_size"]
    cxs = (bed["x"][0] + a + 10, bed["x"][1] - a - 10)
    cys = (bed["y"][0] + a + 10, bed["y"][1] - a - 10)
    loc_tilts = [0, 30, 45, 60] if not args.quick else [0, 45]
    summary["locations"] = {}
    for cx, xl in zip(cxs, ("-x", "+x")):
        for cy, yl in zip(cys, ("-y", "+y")):
            key = f"{xl}{yl} ({cx:.0f},{cy:.0f})"
            summary["locations"][key] = {}
            cfg_l = load_config(planner_overrides={"toolpath": {"patch_center_xy": [cx, cy]}})
            pl_l = Planner(robot, cfg_l, planner.coll)
            for name in SURFACES[:-1]:
                summary["locations"][key][name] = {}
                for t in loc_tilts:
                    tp = make_toolpath(cfg_l, name, t)
                    res = pl_l.plan(tp)
                    s = summarize(res, diagnose(pl_l, tp, res))
                    summary["locations"][key][name][str(t)] = {
                        "planned_pct": s["planned_pct"], "complete": s["complete"],
                        "segments": s["segments"], "breaks": len(s["breaks"]),
                        "reasons": sorted({x["reason"] for x in s["stretches"]}),
                        "runtime_s": s["runtime_s"]["total_s"]}
                    print(f"[loc {key} {name} tilt {t}] {s['planned_pct']:.1f}% segs {s['segments']}", flush=True)

    # Risk 1: continuous turn about the nozzle axis over the bed
    t0 = time.perf_counter()
    xs = np.arange(bed["x"][0] + 20, bed["x"][1], 40.0)
    ys = np.arange(bed["y"][0] + 20, bed["y"][1], 40.0)
    z = bed["top_z"] + cfg.planner["toolpath"]["base_height"]
    W = {}
    for label, d in [("vertical", np.array([0, 0, -1.0]))] + [
            (f"tilt30_{az}", np.array([np.sin(np.radians(30)) * np.cos(np.radians(a)),
                                       np.sin(np.radians(30)) * np.sin(np.radians(a)),
                                       -np.cos(np.radians(30))]))
            for az, a in (("+x", 0), ("-x", 180), ("+y", 90), ("-y", -90))]:
        grid = np.zeros((len(ys), len(xs)))
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                grid[iy, ix] = turn_window(planner, np.array([x, y, z + 20 * (label != "vertical")]), d)[0]
        W[label] = grid
    outputs.plot_turn_window(out / "turn_window.png", xs, ys, W, cfg)
    summary["turn_window"] = {k: {"min": float(v.min()), "median": float(np.median(v[v > 0])) if (v > 0).any() else 0.0,
                                  "max": float(v.max()), "zero_pct": float(100 * (v == 0).mean())}
                              for k, v in W.items()}
    summary["turn_window_center"] = turn_window(
        planner, np.array([*cfg.planner["toolpath"]["patch_center_xy"], z]), np.array([0, 0, -1.0]))
    summary["turn_window_runtime_s"] = round(time.perf_counter() - t0, 1)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("turn window", summary["turn_window"], summary["turn_window_center"])


if __name__ == "__main__":
    main()

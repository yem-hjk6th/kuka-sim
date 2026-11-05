"""Corner test: Hilbert curve layer with a per-corner speed ratio.

Simulation only. Nothing here connects to a robot or a controller.

    python scripts/run_hilbert_corner.py                       # defaults from config/planner.yaml
    python scripts/run_hilbert_corner.py --order 3 --seed 2 --min-straight 20
    python scripts/run_hilbert_corner.py --side 300 --center 700 -410 --out out/hc

Exit code 1 if a check fails (the files are still written, marked as failed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kukasim import hilbert_corner as hc  # noqa: E402
from kukasim.collision import CollisionModel  # noqa: E402
from kukasim.config import load_config  # noqa: E402
from kukasim.first_ptp import StartPoseError  # noqa: E402
from kukasim.kinematics import Robot  # noqa: E402


def main(argv=None, cfg=None):
    """cfg: a loaded config (default: load_config()), e.g. with planner overrides in tests."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/hilbert_corner")
    ap.add_argument("--order", type=int)
    ap.add_argument("--side", type=float, help="side length L in mm (default: fitted to the reachable bed)")
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--ratios", type=float, nargs="+")
    ap.add_argument("--min-repeats", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--min-straight", type=float, help="minimum straight between override windows, mm")
    ap.add_argument("--cdis", type=float, help="C_DIS in mm")
    ap.add_argument("--straight-speed", type=float, help="mm/s (default: process.print_speed_mm_s)")
    ap.add_argument("--straight-override", type=float, help="pc mode: $OV_PRO on straights, %%")
    ap.add_argument("--speed-mode", choices=["krl", "pc"],
                    help="krl (default): $VEL.CP per corner zone in the program; pc: $OV_PRO per schedule.csv "
                         "(needs a PC-side follower, not provided here)")
    ap.add_argument("--d-zone", type=float, help="krl mode: corner zone half-length, mm (>= 2 x C_DIS)")
    ap.add_argument("--dip", type=float, help="corner dip factor of the speed model (1.0 = no dip)")
    args = ap.parse_args(argv)

    cfg = cfg if cfg is not None else load_config()
    robot = Robot.from_config(cfg)
    ct = hc.generate(cfg, robot, order=args.order, side_mm=args.side, center_xy=args.center,
                     ratios=args.ratios, min_repeats=args.min_repeats, seed=args.seed,
                     min_straight_mm=args.min_straight, c_dis_mm=args.cdis,
                     straight_speed_mm_s=args.straight_speed, straight_override_pct=args.straight_override,
                     speed_mode=args.speed_mode, d_zone_mm=args.d_zone, corner_dip_factor=args.dip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hc.write_waypoints_csv(out / "waypoints.csv", ct, robot)
    hc.write_schedule_csv(out / "schedule.csv", ct)
    hc.write_speed_csv(out / "speed_profile.csv", ct)
    src = out / "hilbert_corner.src"
    try:
        hc.write_krl(src, ct, robot, cfg)
        src_note = "KRL export (NOT RUN)"
    except StartPoseError as e:
        # invalid declared start pose: the failed check goes into report.md, no program is written
        src.unlink(missing_ok=True)
        src_note = f"NOT written: first PTP check failed ({e})"
    hc.plot_path(out / "path.png", ct, cfg, CollisionModel(robot, cfg))
    hc.plot_speed(out / "speed.png", ct)
    files = [("waypoints.csv", "start, corners and end with corner id, ratio, segment lengths, joints, "
              "predicted speeds (model) and model-only bead width"),
             ("schedule.csv", "one row per corner: zones and corner speed (pc mode: the $OV_PRO schedule for a "
              "PC-side follower, not provided in this repository)"),
             ("speed_profile.csv", "model speed along the path"),
             ("hilbert_corner.src", src_note),
             ("path.png", "path with corner ids and ratios"),
             ("speed.png", "model speed profile")]
    hc.write_report(out / "report.md", ct, cfg, files)
    n_c = sum(k == "corner" for k in ct.kinds)
    print(f"order {ct.params['order']}, side {ct.side:g} mm, centre {ct.center}, {n_c} corners, "
          f"{int(ct.excluded.sum())} set to r = 1")
    for c in ct.checks:
        tag = "ok  " if c["ok"] else ("FAIL" if c["severity"] == "error" else "warn")
        print(f"  [{tag}] {c['name']}: {c['value']} (limit {c['limit']}) {c['detail'] if not c['ok'] else ''}")
    print(f"written to {out}/")
    return 0 if ct.ok else 1


if __name__ == "__main__":
    sys.exit(main())

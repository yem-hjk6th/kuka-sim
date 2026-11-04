"""Write RESULTS.md from results/summary.json (produced by run_step1.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


FINDINGS = """## Findings

- **Bed centre: no limit from tilt.** All four surfaces plan as one continuous segment for every tilt cap
  from 0 to 60 deg (that is, up to the full surface normal, 38-46 deg). At the default cap the joint-limit margin stays >= 19 deg
  and |A5| >= 28 deg. Runtime is 10-15 s per path of ~1400 points on one CPU core
  (about 70 % building states (IK + collision), 30 % DP).
- **Toward the edges, the clamps decide, not the tilt.** Every gap in the quadrant runs is a clamp
  collision, except one short out-of-reach stretch near the +X/-Y corner (dome). Raising the cap
  from 0 to 45 deg costs up to 9 % more of the path (dome at -X), because a tilted tool body sweeps
  further sideways into the clamp boxes.
- **The -X clamp pair blocks a band along the whole -X edge** because its Y positions are unknown, so it is
  modelled as a strip over the full edge. The -X/-Y and -X/+Y results are identical for this reason.
  Measuring the Y of those two clamps would free most of that band.
- **The perimeter path (bed outline inset 40 mm) is 56 % plannable**, cut into 3 pieces at the
  three modelled clamp locations.
- **The DP sometimes turns the nozzle a lot.** On the dome it rotates by 215 deg about the nozzle axis
  within 75 mm of path (s = 1230-1305 mm, at most 5 deg per waypoint, A4 step <= 5.2 deg). This is
  inside every hard limit and cheap under the joint-space smoothness cost, but it would wind the hose.
  A cost on the redundancy angle's rate, or a cap on it, is the obvious knob (`max_dbin`, weights in
  `config/planner.yaml`).
- **On flat or gently curved paths the chosen angle is nearly constant** (sine 90-100 deg, ramp 90 deg:
  tool +Z towards BASE -Y), away from the 0 deg of the planar programs (A0 B90 C0). With the default weights
  any angle well inside the limits costs almost the same, so this choice is arbitrary to within the window.
"""


def fmt_tilts(d):
    ok = [int(t) for t, v in d.items() if v["complete"]]
    allt = [int(t) for t in d]
    if not ok:
        return "none"
    if ok == allt:
        return f"{min(ok)}-{max(ok)} deg (all tested)"
    return ", ".join(str(t) for t in ok) + " deg"


def transit_section(s):
    """Before / after numbers for the moves between printed segments (issue #2)."""
    ex = {n: v["export"] for n, v in s["surfaces"].items() if "export" in v}
    if not ex:
        return []
    p = next(iter(ex.values()))["params"]
    L = ["## KRL export: moves between segments (issue #2)\n"]
    L.append("Before, a segment break in the export was a PTP straight from the last printed point to the next "
             "start, at print height, unchecked, at the controller's default axis speed. Now each break is a LIN "
             f"retract along the nozzle axis to at least {p['safe_height_mm']} mm above the bed top (and at least "
             f"{p['printed_clearance_mm']} mm above the highest Z printed so far), a PTP that is "
             "sampled in joint space and checked against the bed, the clamps and the floor (tip included), lifted to "
             f"{', '.join(str(h) for h in p['via_heights_mm'])} mm if it collides, and a LIN approach; if no candidate "
             f"is clear, the export fails. `$VEL_AXIS` = {p['vel_axis_pct']} % and `$ACC_AXIS` = {p['acc_axis_pct']} % "
             "are set on all axes; before every transit PTP `$VEL_AXIS` is lowered to "
             f"floor({p['vel_axis_pct']} x min(1, {p['speed_safety_factor']} x {p['max_tcp_speed_mm_s']} / peak)) % "
             f"(at least 1 %) so the model peak is <= {p['enforced_tcp_speed_mm_s']:g} mm/s; a transit still above "
             "that at 1 % fails the export unless `allow_fast_transit` is set. A transit that turns A4 or A6 by more "
             f"than {p['max_wrist_turn_deg']} deg is flagged (hose / cable wrap not modelled).\n")
    L.append("TCP speeds are a **model estimate**: every axis at most `$VEL_AXIS` % of its URDF velocity "
             "(unverified) and `$ACC_AXIS` % of an assumed acceleration, the leading axis sets a trapezoid rate "
             "along the joint-space line (a move too short to reach cruise is capped by its distance), peak TCP "
             "speed = max |dx/ds| x that rate. 'Before' assumes 100 % axis speed and acceleration. Clearance is the margin to the inflated "
             "collision boxes (bed gap 5 mm, clamps +20 mm); negative means inside. The bed top is at Z "
             f"{p['bed_top_z']:.2f}.\n")
    L.append("| toolpath | transit | before: min clearance (obstacles hit) | before: lowest TCP Z | "
             "before: peak TCP (model) | after: PTP height | after: min clearance per move | "
             "after: min distance to bed | after: $VEL_AXIS | after: peak TCP (model) | rejected candidates | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    n = 0
    for name, e in ex.items():
        for b, t in zip(e["before_direct_ptp"], e["transits"]):
            n += 1
            rej = "; ".join(f"z {r['z_mm']:.0f}: {r['reason']}" for r in t["rejected"]) or "-"
            L.append(f"| {name} | {t['from_segment']}->{t['to_segment']} | {b['min_clearance_mm']:.1f} mm "
                     f"({', '.join(b['hits']) or 'clear'}) | Z {b['min_tcp_z_mm']:.1f} | {b['peak_tcp_mm_s_model']:.0f} mm/s | Z {t['z_mm']:.1f} | "
                     + ", ".join(f"{k} {v:.1f}" for k, v in t["move_min_clearance_mm"].items())
                     + f" mm | {t['min_bed_distance_mm']:.1f} mm | {t['ptp_vel_axis_pct']:g} % | "
                     f"{t['peak_tcp_mm_s_model']:.0f} mm/s | {rej} | {'; '.join(t['flags']) or 'none'} |")
    if not n:
        L.append("| - | no segment breaks | | | | | | | | | | |")
    L.append("")
    L.append("Clearance per move: retract / approach LINs with the tip exempt from the bed (they start on the "
             "print), PTP and lift / lower LINs with the tip included.\n")
    L.append(f"Flag thresholds: peak TCP speed > {p['enforced_tcp_speed_mm_s']:g} mm/s (only with "
             f"`allow_fast_transit`), |dA4| or |dA6| > {p['max_wrist_turn_deg']} deg, any collision, any primitive "
             f"(tip included) closer than {p['bed_clearance_mm']} mm to the bed at transit height. Retract and "
             "approach LINs run at the print `$VEL.CP` with the orientation held. Every export also approaches its "
             "first point and leaves its last point the same way.\n")
    fps = {n: v["export"]["first_ptp"] for n, v in s["surfaces"].items() if "first_ptp" in v.get("export", {})}
    if fps:
        f0 = next(iter(fps.values()))
        r = ", ".join(f"{v:g}" for v in f0["axis_tip_radii_mm"])
        L.append("First PTP (from the pose the robot is in when a program starts; `kukasim/first_ptp.py`): its "
                 "TCP speed is at most sum_i ($VEL_AXIS % x URDF axis max speed) x r_i from any start, with the "
                 f"axis-to-tool-tip radii r_i = {r} mm (bounded from the URDF and TOOL 3). At $VEL_AXIS 1 % this "
                 f"bound is {f0['bound_at_1pct_mm_s']:g} mm/s, "
                 + (f"<= {f0['limit_mm_s']:g} mm/s, so the programs set $VEL_AXIS {f0['vel_axis_pct']} % "
                    "(case `bound`)." if f0["case"] == "bound" else
                    f"above {f0['limit_mm_s']:g} mm/s, so no start-independent bound exists (case `declared start`): "
                    "the programs require the robot to be at the declared start pose "
                    f"{{{', '.join(f'A{i + 1} {v:g}' for i, v in enumerate(f0['start_axis_deg']))}}} "
                    f"(+-{f0['start_tolerance_deg']:g} deg per axis, `transit.start_axis`, a default to be replaced "
                    "by the cell's real start pose) and lower $VEL_AXIS for the first PTP with the transit speed "
                    "model. The path of the first PTP is not collision-checked.") + "\n")
        L.append("| surface | case | first PTP $VEL_AXIS | peak TCP (model, worst over the tolerance box) |\n"
                 "|---|---|---|---|")
        for n, f in fps.items():
            L.append(f"| {n} | {f['case']} | {f['vel_axis_pct']} % | {f['peak_tcp_mm_s']:g} mm/s |")
        L.append("")
    return L


def main(res_dir="results"):
    rd = REPO / res_dir
    s = json.loads((rd / "summary.json").read_text())
    L = []
    L.append("# Step 1 results: DP baseline for the redundancy angle\n")
    L.append("Simulation only. Generated by `python scripts/run_step1.py --out results` and "
             "`python scripts/write_results.py`. Every number depends on the estimated "
             "parameters in `config/planner.yaml` (tool and link capsules, clamp boxes, bed gap) and on the "
             "`estimated`/`unknown` values in `config/cell.yaml`.\n")
    L.append(f"Default nozzle tilt cap: {s['default_max_tilt_deg']} deg. Waypoints every 1.5 mm, "
             "72 redundancy bins (5 deg), 32 IK slots per bin. Test surfaces are 240 x 240 mm patches centred "
             "on the bed (713, -410), planned on the layer 10 mm above the bed top; rasters 40 mm apart.\n")

    L.append("## Per surface (default tilt cap, bed centre)\n")
    L.append("| surface | points / length | planned | segments | max surface tilt | redundancy angle used | "
             "min joint-limit margin* | min abs(A5) | min clearance | runtime (states + DP) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, v in s["surfaces"].items():
        rt = v["runtime_s"]
        th = v["theta_range_deg"]
        L.append(f"| {n} | {v['points']} / {v['length_mm']:.0f} mm | {v['planned_pct']:.1f} % | {v['segments']} | "
                 f"{v['max_normal_tilt_deg']:.0f} deg | "
                 f"{'%.0f..%.0f deg (span %.0f)' % (th[0], th[1], v['theta_span_deg']) if th else '-'} | "
                 f"{v['min_limit_margin_deg']:.1f} deg | {v['min_abs_a5_deg']:.1f} deg | "
                 f"{v['min_clearance_mm']:.1f} mm | {rt['total_s']:.1f} s ({rt['states_s']:.1f} + {rt['dp_s']:.1f}) |")
    L.append("\n*beyond the 2 deg margin from cell.yaml.\n")

    L.append("### Infeasible stretches (default tilt cap)\n")
    any_st = False
    for n, v in s["surfaces"].items():
        for st in v["stretches"]:
            any_st = True
            a, b = st["points"]
            L.append(f"- **{n}**: points {a}-{b} (s = {st['s_mm'][0]:.0f}-{st['s_mm'][1]:.0f} mm, "
                     f"around XY {st['xy_mid']}): {st['reason']}")
    if not any_st:
        L.append("- none")
    L.append("")

    L.append("## Reachable tilt range\n")
    L.append("A tilt cap counts as reachable when the whole path is planned in one continuous segment. "
             "The caps only bite up to each surface's own maximum normal tilt (dome 46, saddle 43, sine 38, ramp 44 deg); "
             "the 60 deg cap therefore means 'follow the normal everywhere'.\n")
    L.append("### Bed centre\n")
    L.append("| surface | reachable caps | runtime per run |")
    L.append("|---|---|---|")
    for n, d in s["sweep"].items():
        rts = [v["runtime_s"]["total_s"] for v in d.values()]
        L.append(f"| {n} | {fmt_tilts(d)} | {min(rts):.1f}-{max(rts):.1f} s |")
    L.append("")
    if "locations" in s:
        L.append("### Same surfaces moved to the bed quadrants\n")
        L.append("Planned share of the path per tilt cap; `x` marks a complete plan. The last column lists the reasons for the gaps. "
             "These patches reach to 10 mm from the bed edges, so they overlap the clamp boxes on purpose.\n")
        tilts = None
        for key, sd in s["locations"].items():
            tilts = list(next(iter(sd.values())).keys())
            break
        L.append("| patch centre | surface | " + " | ".join(f"cap {t}" for t in tilts) + " | reasons |")
        L.append("|---|---|" + "---|" * len(tilts) + "---|")
        for key, sd in s["locations"].items():
            for n, d in sd.items():
                cells = [("x " if v["complete"] else "") + f"{v['planned_pct']:.0f} %" for v in d.values()]
                reasons = sorted({r for v in d.values() for r in v["reasons"]})
                L.append(f"| {key} | {n} | " + " | ".join(cells) + f" | {'; '.join(reasons) or '-'} |")
        L.append("")

    L.append(FINDINGS)
    tw = s["turn_window"]
    L.append("## Continuous turn about the nozzle axis\n")
    L.append("Longest rotation about the nozzle axis that can be swept continuously at a fixed point "
             "(5 deg bins, at most 10 deg per joint between bins, limits with margin, collisions), "
             f"on a 40 mm grid over the bed. Bed centre, vertical nozzle: {s['turn_window_center'][0]:.0f} deg.\n")
    L.append("`tilt30 +x` means the nozzle (tool +X) is tilted 30 deg from straight down towards BASE +X, "
             "i.e. the tip points away from the robot. Values over 360 deg use the second A6 turn.\n")
    L.append("At the bed centre the window is set by A5 = +-120 deg (+-118 with the margin): with A5 widened "
             "to +-180 and collisions off it becomes a full 360 deg. Collisions do not shrink it there. "
             "This matches the earlier 235-285 deg.\n")
    L.append("| nozzle | median | max | grid points with no feasible angle |")
    L.append("|---|---|---|---|")
    for k, v in tw.items():
        L.append(f"| {k.replace('_', ' ')} | {v['median']:.0f} deg | {v['max']:.0f} deg | {v['zero_pct']:.0f} % |")
    L.append("\n![turn window](results/turn_window.png)\n")

    L += transit_section(s)

    L.append("## Files per toolpath (`results/<surface>/`)\n")
    L.append("`trajectory.csv` (joints, TCP, ABC, margins), `joints.png`, `margins.png`, `feasibility.png`, "
             "`animation.gif`, `<surface>.src` (KRL, NOT RUN header), `export_report.json` (transits and their flags).\n")
    (REPO / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""KRL export of a planned trajectory. Simulation only: every file starts with the NOT RUN header.

Segments are joined by planned and checked transits (kukasim/transit.py, issue #2):
LIN retract along the nozzle axis, PTP at a safe height (lifted further if needed),
LIN approach. If no clear transit exists the export fails with TransitError.
"""
from __future__ import annotations

import re

import numpy as np

from .transit import TransitPlanner, end_moves, plan_transits

NOT_RUN = "; NOT RUN - generated in simulation, never executed on a robot"


def krl_name(name):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return n[:24]


def _lin(p, abc, cdis=False):
    return "  LIN {X %.3f, Y %.3f, Z %.3f, A %.4f, B %.4f, C %.4f}" % (*p, *abc) + (" C_DIS" if cdis else "")


def _ptp(q):
    return "  PTP {A1 %.4f, A2 %.4f, A3 %.4f, A4 %.4f, A5 %.4f, A6 %.4f}" % tuple(np.degrees(q))


def _emit_move(lines, m, abc):
    """Write one planned move; LIN moves hold the orientation abc."""
    if m.kind == "PTP":
        lines.append(_ptp(m.q[-1]))
    else:
        lines.append(_lin(m.tcp[-1], abc))


def write_krl(path, robot, res, rows, cfg, name, tplanner=None):
    """rows: from outputs.tcp_rows (TCP position + ABC per point, None if infeasible).

    Returns the export report (dict): transit parameters, every transit with its
    checks and flags, and the pre-fix direct PTP for comparison."""
    tool = cfg.tool_xyzabc
    tpl = tplanner or TransitPlanner.from_config(robot, cfg)
    tp = tpl.p
    transits = plan_transits(tpl, res, rows)          # raises TransitError if one is not clear
    first, last = end_moves(tpl, res, rows)
    before = []
    for k in range(1, len(res.segments)):
        d = tpl.direct_ptp(res.q[res.segments[k - 1][1]], res.q[res.segments[k][0]])
        before.append({"from_segment": k - 1, "to_segment": k,
                       "min_clearance_mm": round(d.clearance_mm, 2), "blocking": d.blocking, "hits": d.hits,
                       "min_tcp_z_mm": round(float(d.tcp[:, 2].min()), 2),
                       "min_bed_distance_mm": round(d.bed_distance_mm, 2),
                       "peak_tcp_mm_s_model": round(d.peak_tcp_mm_s, 1),
                       "start_z_mm": round(float(d.tcp[0][2]), 2), "end_z_mm": round(float(d.tcp[-1][2]), 2),
                       "vel_axis_pct_assumed": tp["before_vel_axis_pct"]})
    report = {
        "toolpath": res.toolpath.name, "segments": [list(map(int, s)) for s in res.segments],
        "params": {k: tp[k] for k in ("safe_height_mm", "min_retract_mm", "via_heights_mm", "bed_clearance_mm",
                                      "vel_axis_pct", "acc_axis_pct", "max_tcp_speed_mm_s")}
        | {"bed_top_z": tpl.bed_top},
        "speed_model": "model estimate: each axis <= vel_axis_pct of its URDF velocity (unverified); "
                       "peak TCP speed <= max|dx/ds| * min_j(v_j/|dq_j|); controller profile not modelled",
        "transits": [t.summary() for t in transits],
        "flags": [f"transit {t.seg_from}->{t.seg_to}: {f}" for t in transits for f in t.flags],
        "before_direct_ptp": before,
    }

    lines = [
        NOT_RUN,
        "; This file was produced by kuka-sim (step 1, DP baseline) and has never been run.",
        "; Do not load it on a controller without a separate review and a dry run.",
        f"; toolpath: {res.toolpath.name}, max tilt {res.toolpath.max_tilt_deg:g} deg, "
        f"{len(res.bins)} points, {int(res.feasible.sum())} planned",
        f"; TOOL 3 assumed = {{X {tool[0]}, Y {tool[1]}, Z {tool[2]}, A {tool[3]}, B {tool[4]}, C {tool[5]}}}"
        " (copied, to be re-confirmed); BASE 0",
        "; Orientation is A/B/C (Rz(A) Ry(B) Rx(C)); B ~ 90 is an Euler singularity, only A - C matters there.",
        "; LIN points carry no S/T; configuration changes happen only in PTP moves given as joint values (E6AXIS).",
        "; Joint values for every point are in trajectory.csv. Controller blending (C_DIS) is not simulated.",
        f"; Transits: LIN retract along the nozzle axis (tool -X) to >= {tp['safe_height_mm']:g} mm above the bed top,",
        f";   PTP (lifted to {', '.join(f'{h:g}' for h in tp['via_heights_mm'])} mm if the low one collides), LIN approach.",
        ";   Every transit was sampled (PTP in joint space) and checked against the bed, clamps and floor in simulation.",
        f"; $VEL_AXIS = {tp['vel_axis_pct']:g} %, $ACC_AXIS = {tp['acc_axis_pct']:g} % on all axes. TCP speeds below are a",
        ";   MODEL ESTIMATE (axes at <= $VEL_AXIS of the unverified URDF velocities), not controller output.",
        "; The first PTP comes from the unknown current robot pose and is NOT checked.",
    ]
    for t in transits:
        s = t.summary()
        lines.append(f"; transit {t.seg_from}->{t.seg_to}: PTP at Z {s['z_mm']:.1f}, min clearance "
                     f"{s['min_clearance_mm']:.1f} mm, bed distance {s['min_bed_distance_mm']:.1f} mm, "
                     f"peak TCP {s['peak_tcp_mm_s_model']:.0f} mm/s (model)")
    lines.append("; FLAGS: " + ("; ".join(report["flags"]) if report["flags"] else "none"))
    lines += [
        f"DEF {krl_name(name)}()",
        "  $BASE = $NULLFRAME",
        "  $TOOL = TOOL_DATA[3]",
        f"  $VEL.CP = {cfg.print_speed / 1000.0:.3f}",
        "  $APO.CDIS = 0.5",
        "  $ORI_TYPE = #VAR",
    ]
    lines += [f"  $VEL_AXIS[{i}] = {tp['vel_axis_pct']:g}" for i in range(1, 7)]
    lines += [f"  $ACC_AXIS[{i}] = {tp['acc_axis_pct']:g}" for i in range(1, 7)]
    in_seg = False
    for k, (a, b) in enumerate(res.segments):
        if k == 0:
            lines.append(f"  ; approach above the start of segment 0 (Z {first.tcp[0][2]:.1f})")
            lines.append(_ptp(first.q[0]) + " ; from the current pose: NOT checked")
            lines.append(_lin(rows[a][0], rows[a][1]))
        else:
            t = transits[k - 1]
            lines.append(f"  ; ---- transit {k - 1}->{k}: checked in simulation, PTP at Z {t.z_mm:.1f}"
                         + (f" (FLAGS: {'; '.join(t.flags)})" if t.flags else ""))
            lines.append("  HALT")
            e = res.segments[k - 1][1]
            for m in t.moves:
                abc = rows[a][1] if m.label in ("lower", "approach") else rows[e][1]
                lines.append(f"  ; {m.kind} {m.label}")
                if m.label == "approach":
                    lines.append(_lin(rows[a][0], rows[a][1]))
                else:
                    _emit_move(lines, m, abc)
        lines.append(f"  ; segment {k}: points {a}..{b}")
        for i in range(a + 1, b + 1):
            p, abc = rows[i]
            lines.append(_lin(p, abc, cdis=i < b))
        in_seg = True
    if last is not None:
        e = res.segments[-1][1]
        lines.append(f"  ; final retract to Z {last.tcp[-1][2]:.1f}")
        lines.append(_lin(last.tcp[-1], rows[e][1]))
    for a, b in res.no_state:
        lines.append(f"  ; INFEASIBLE: points {a}..{b} have no feasible state (omitted)")
    if not in_seg:
        lines.append("  ; no feasible point on this toolpath")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report

"""KRL export of a planned trajectory. Simulation only: every file starts with the NOT RUN header.

Segments are joined by planned and checked transits (kukasim/transit.py, issue #2):
LIN retract along the nozzle axis, PTP at a safe height (lifted further if needed),
LIN approach. If no clear transit exists the export fails with TransitError.
Every transit PTP is preceded by its own $VEL_AXIS[1..6], lowered so the model peak
TCP speed stays <= speed_safety_factor x max_tcp_speed_mm_s. Files are written with
LF line endings on every platform.

The first PTP starts from the pose the robot is in; its $VEL_AXIS and the header
lines about it come from kukasim/first_ptp.py (start-independent bound, or a declared
start pose when no integer $VEL_AXIS >= 1 % is slow enough from every start).

Numbers are written in one fixed form so committed results do not change on
last-bit noise: fixed decimals, never a negative zero, and orientation angles
A / C in (-180, 180] (a value that rounds to -180 is written as 180). Joint angles
are not wrapped (A4 / A6 span more than 360 deg).
"""
from __future__ import annotations

import math
import re

import numpy as np

from .first_ptp import CASE_NONE, plan_first_ptp
from .transit import TransitPlanner, end_moves, plan_transits

NOT_RUN = "; NOT RUN - generated in simulation, never executed on a robot"


def krl_name(name):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return n[:24]


def fnum(x, nd):
    """x with nd fixed decimals; never a negative zero ("-0.0000" -> "0.0000")."""
    s = f"{float(x):.{nd}f}"
    return f"{0.0:.{nd}f}" if float(s) == 0.0 else s


def fang(x, nd):
    """Orientation angle (deg) with nd fixed decimals, wrapped to (-180, 180]: a value that
    rounds to -180 is written as 180, so +-180 noise always gives the same text."""
    s = fnum(math.remainder(float(x), 360.0), nd)
    return fnum(180.0, nd) if float(s) <= -180.0 else s


def no_negative_zero(obj):
    """Copy of a report (dicts / lists / numbers) with every -0.0 replaced by 0.0."""
    if isinstance(obj, dict):
        return {k: no_negative_zero(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [no_negative_zero(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and obj == 0.0:
        return 0.0
    return obj


def lin_pose(p, abc):
    """{X .., Y .., Z .., A .., B .., C ..} of a LIN target (mm 3 decimals, deg 4 decimals)."""
    return (f"{{X {fnum(p[0], 3)}, Y {fnum(p[1], 3)}, Z {fnum(p[2], 3)}, "
            f"A {fang(abc[0], 4)}, B {fnum(abc[1], 4)}, C {fang(abc[2], 4)}}}")


def ptp_axes(q_deg):
    """{A1 .., ..., A6 ..} of a PTP target given in degrees (4 decimals, not wrapped)."""
    return "{" + ", ".join(f"A{i + 1} {fnum(v, 4)}" for i, v in enumerate(q_deg)) + "}"


def _lin(p, abc, cdis=False):
    return "  LIN " + lin_pose(p, abc) + (" C_DIS" if cdis else "")


def _ptp(q):
    return "  PTP " + ptp_axes(np.degrees(q))


def _emit_move(lines, m, abc):
    """Write one planned move; LIN moves hold the orientation abc. A PTP that carries
    its own axis speed gets $VEL_AXIS[1..6] right before it."""
    if m.kind == "PTP":
        if m.vel_axis_pct is not None:
            lines += [f"  $VEL_AXIS[{i}] = {m.vel_axis_pct:g}" for i in range(1, 7)]
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
    fp = plan_first_ptp(robot, cfg, None if first is None else first.q[0], tp["acc_axis_pct"])
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
        "params": {k: tp[k] for k in ("safe_height_mm", "min_retract_mm", "printed_clearance_mm", "via_heights_mm",
                                      "bed_clearance_mm", "vel_axis_pct", "acc_axis_pct", "max_tcp_speed_mm_s",
                                      "speed_safety_factor", "allow_fast_transit", "axis_acc_deg_s2",
                                      "max_wrist_turn_deg")}
        | {"bed_top_z": tpl.bed_top, "enforced_tcp_speed_mm_s": tpl.speed_limit},
        "speed_model": "model estimate: each axis <= $VEL_AXIS % of its URDF velocity (unverified) and "
                       "$ACC_AXIS % of the assumed axis_acc_deg_s2; path rate = trapezoid in s "
                       "(cruise min_j(v_j/|dq_j|), ramps min_j(a_j/|dq_j|)); peak TCP speed = max |dx/ds| * rate; "
                       "controller profile not modelled",
        "clearance_note": "move_min_clearance_mm: retract / approach LINs are checked with the tip exempt from "
                          "the bed (they start on the print); PTP and lift / lower LINs with the tip included",
        "transits": [t.summary() for t in transits],
        "flags": [f"transit {t.seg_from}->{t.seg_to}: {f}" for t in transits for f in t.flags]
                 + ([] if fp.ok else [f"first PTP: case '{fp.case}', not checked" if fp.case == CASE_NONE
                                      else f"first PTP: case '{fp.case}' above the limit"]),
        "before_direct_ptp": before,
        "first_ptp": fp.summary(),
    }

    report = no_negative_zero(report)
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
        f";   and >= {tp['printed_clearance_mm']:g} mm above the highest Z printed so far,",
        f";   PTP (lifted to {', '.join(f'{h:g}' for h in tp['via_heights_mm'])} mm if the low one collides), LIN approach.",
        ";   Every transit was sampled (PTP in joint space) and checked against the bed, clamps and floor in simulation.",
        f"; $VEL_AXIS = {tp['vel_axis_pct']:g} %, $ACC_AXIS = {tp['acc_axis_pct']:g} % on all axes; before each transit PTP",
        f";   $VEL_AXIS is lowered so its peak TCP speed is <= {tpl.speed_limit:g} mm/s "
        f"(= {tp['speed_safety_factor']:g} x {tp['max_tcp_speed_mm_s']:g}).",
        ";   TCP speeds below are a MODEL ESTIMATE (axes at <= $VEL_AXIS of the unverified URDF velocities,",
        ";   assumed accelerations), not controller output. Clearances: retract / approach with the tip exempt",
        ";   from the bed (they start on the print), PTP and lift / lower with the tip included.",
        "; The path of the first PTP (from the pose the robot is in) is NOT collision-checked.",
    ]
    lines += fp.header()
    for t in transits:
        s = t.summary()
        per = ", ".join(f"{k} {fnum(v, 1)}" for k, v in s["move_min_clearance_mm"].items())
        lines.append(f"; transit {t.seg_from}->{t.seg_to}: PTP at Z {fnum(s['z_mm'], 1)}, min clearance per move (mm): "
                     f"{per}; bed distance {fnum(s['min_bed_distance_mm'], 1)} mm, $VEL_AXIS {s['ptp_vel_axis_pct']:g} %, "
                     f"peak TCP {fnum(s['peak_tcp_mm_s_model'], 0)} mm/s (model)")
    lines.append("; FLAGS: " + ("; ".join(report["flags"]) if report["flags"] else "none"))
    lines += [
        f"DEF {krl_name(name)}()",
        "  $BASE = $NULLFRAME",
        "  $TOOL = TOOL_DATA[3]",
        f"  $VEL.CP = {cfg.print_speed / 1000.0:.3f}",
        "  $APO.CDIS = 0.5",
        "  $ORI_TYPE = #VAR",
    ]
    lines += [f"  $VEL_AXIS[{i}] = {fp.vel_axis_pct:d}" for i in range(1, 7)]   # first PTP (first_ptp.py)
    lines += [f"  $ACC_AXIS[{i}] = {tp['acc_axis_pct']:g}" for i in range(1, 7)]
    in_seg = False
    for k, (a, b) in enumerate(res.segments):
        if k == 0:
            lines.append(f"  ; approach above the start of segment 0 (Z {fnum(first.tcp[0][2], 1)})")
            lines.append(_ptp(first.q[0]) + (" ; first PTP: speed bounded from any start pose" if fp.case == "bound"
                                             else " ; first PTP: from the DECLARED START pose (see header)"))
            lines.append(_lin(rows[a][0], rows[a][1]))
        else:
            t = transits[k - 1]
            lines.append(f"  ; ---- transit {k - 1}->{k}: checked in simulation, PTP at Z {fnum(t.z_mm, 1)}"
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
        lines.append(f"  ; final retract to Z {fnum(last.tcp[-1][2], 1)}")
        lines.append(_lin(last.tcp[-1], rows[e][1]))
    for a, b in res.no_state:
        lines.append(f"  ; INFEASIBLE: points {a}..{b} have no feasible state (omitted)")
    if not in_seg:
        lines.append("  ; no feasible point on this toolpath")
    lines.append("END")
    with open(path, "w", newline="\n") as f:          # LF on every platform (K-2)
        f.write("\n".join(lines) + "\n")
    return report

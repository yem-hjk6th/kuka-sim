"""KRL export of a planned trajectory. Simulation only: every file starts with the NOT RUN header."""
from __future__ import annotations

import re

import numpy as np

NOT_RUN = "; NOT RUN - generated in simulation, never executed on a robot"


def krl_name(name):
    n = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return n[:24]


def write_krl(path, robot, res, rows, cfg, name):
    """rows: from outputs.tcp_rows (TCP position + ABC per point, None if infeasible)."""
    tool = cfg.tool_xyzabc
    lines = [
        NOT_RUN,
        "; This file was produced by kuka-sim (step 1, DP baseline) and has never been run.",
        "; Do not load it on a controller without a separate review and a dry run.",
        f"; toolpath: {res.toolpath.name}, max tilt {res.toolpath.max_tilt_deg:g} deg, "
        f"{len(res.bins)} points, {int(res.feasible.sum())} planned",
        f"; TOOL 3 assumed = {{X {tool[0]}, Y {tool[1]}, Z {tool[2]}, A {tool[3]}, B {tool[4]}, C {tool[5]}}}"
        " (copied, to be re-confirmed); BASE 0",
        "; Orientation is A/B/C (Rz(A) Ry(B) Rx(C)); B ~ 90 is an Euler singularity, only A - C matters there.",
        "; LIN points carry no S/T; each continuous segment starts with an axis-specific PTP (E6AXIS).",
        "; Joint values for every point are in trajectory.csv. Controller blending (C_DIS) is not simulated.",
        f"DEF {krl_name(name)}()",
        "  $BASE = $NULLFRAME",
        "  $TOOL = TOOL_DATA[3]",
        f"  $VEL.CP = {cfg.print_speed / 1000.0:.3f}",
        "  $APO.CDIS = 0.5",
        "  $ORI_TYPE = #VAR",
    ]
    qd = np.degrees(res.q)
    in_seg = False
    for k, (a, b) in enumerate(res.segments):
        if k > 0:
            lines.append(f"  ; ---- segment break before point {a}: reconfiguration NOT planned or collision-checked")
            lines.append("  HALT")
        q = qd[a]
        lines.append(f"  ; segment {k}: points {a}..{b}")
        lines.append("  PTP {A1 %.4f, A2 %.4f, A3 %.4f, A4 %.4f, A5 %.4f, A6 %.4f}" % tuple(q))
        for i in range(a, b + 1):
            p, abc = rows[i]
            lines.append("  LIN {X %.3f, Y %.3f, Z %.3f, A %.4f, B %.4f, C %.4f} C_DIS" % (*p, *abc))
        in_seg = True
    for a, b in res.no_state:
        lines.append(f"  ; INFEASIBLE: points {a}..{b} have no feasible state (omitted)")
    if not in_seg:
        lines.append("  ; no feasible point on this toolpath")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

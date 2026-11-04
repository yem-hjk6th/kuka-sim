"""Static check of a KRL program (.src) against the cell model. Simulation only.

The program is read line by line (see ``parse_program``). Only an explicit list of statements is
accepted: motions (PTP / LIN), the modelled settings and frames, and statements without motion
(DECL, call-free assignments to user variables, WAIT SEC, HALT, END, RSI_CREATE / ON / OFF,
$OV_PRO, PDAT_ACT, $BWDSTART, $ADVANCE, call-free IF / SWITCH / FOR / WHILE / LOOP / REPEAT).
Every other statement, including any call of a function or subroutine, is "not interpreted"
(exit code 2), and every setting it may change becomes unknown. Branches and loops are followed
by abstract interpretation: each setting holds the set of values it may have, and a motion is
checked with the worst case. Every motion is then replayed on the models that already exist in
this repository, nothing is re-modelled here:

* kinematics: ``kinematics.Robot`` (URDF FK, analytic IK, TOOL 3 / BASE 0 of cell.yaml);
* collision: ``collision.CollisionModel`` (capsules, bed / clamp / keep-out boxes, floor);
* PTP sampling and the PTP TCP-speed estimate: ``transit.TransitPlanner``.

Checks per motion (thresholds: ``src_check`` in config/planner.yaml):

1. reach: IK inside the URDF joint limits minus ``use_margin_deg``; a LIN is sampled and
   one IK branch is followed from its start: any configuration (status) change, joint
   step above ``lin_max_joint_step_deg`` or sample without IK is flagged;
2. singularity margins (|A5|, elbow, shoulder) on every sample;
3. speed (M1): LIN at the highest ``$VEL.CP`` that may be in force; PTP on the cruise bound
   (synchronous PTP, leading axis at its ``$VEL_AXIS``; independent of the assumed axis
   accelerations), with the ramp-limited peak reported for information; always at 100 % override;
4. clearance (M2): minimum signed clearance and its primitive / obstacle pair; the nozzle
   tip is exempt from the bed on LIN moves only;
5. nozzle height (M3): tip Z above the highest bed top, and tip XY of LIN moves inside the
   bed outline minus a margin.

A motion whose start pose or settings are unknown is "not checked" (target pose only). Each DEF of a file is
its own section; a DEF after the first one starts with an unknown pose and unknown settings.
Exit codes: 0 every motion checked and no flag; 1 at least one flag; 2 parse / frame error or
a statement not interpreted; 3 no flag but at least one motion not checked.
"No flag" means no flag in this model; it does not replace a check on the controller.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from .collision import CollisionModel, keep_out_boxes
from .config import REPO, load_config
from .kinematics import Robot, xyzabc_to_T
from .rotations import make_transform, rotation_angle_between
from .transit import TransitPlanner

MODELLED_TOOL = 3
MODELLED_BASE = 0
LOCAL_KEEP_OUT = REPO / "config" / "keep_out.local.yaml"

# issue codes; severity "error" -> exit 2, "flag" -> exit 1
ERRORS = ("parse_error", "frame_not_modelled", "not_interpreted", "invalid_setting")
FLAGS = ("reach", "lin_branch", "singularity", "speed", "clearance", "nozzle_height", "bed_edge")
EXIT_RESULT = {0: "no flag, every motion checked (in this model)", 1: "flagged",
               2: "parse / frame error or statement not interpreted", 3: "no flag, but motion(s) not checked"}

MOTION_WORDS = ("PTP", "LIN", "CIRC", "SPTP", "SLIN", "SCIRC", "PTP_REL", "LIN_REL", "CIRC_REL",
                "SPTP_REL", "SLIN_REL", "SCIRC_REL", "SPL", "SPLINE", "ASYPTP")

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE = {k: re.compile(v, re.IGNORECASE) for k, v in {
    "attribute": r"^&\w+",
    "def": r"^(?:GLOBAL\s+)?DEF\s+(\w+)\s*\([^)]*\)$",
    "end": r"^END$",
    "halt": r"^HALT$",
    "wait": rf"^WAIT\s+SEC\s+(?:{_NUM}|[A-Za-z_]\w*)$",
    "decl": r"^DECL\s+\w+\s+\w+(.*)$",
    # control flow (conditions must be call-free, see _pure)
    "if": r"^IF\s+(.+)\s+THEN$",
    "else": r"^ELSE$",
    "endif": r"^ENDIF$",
    "switch": r"^SWITCH\s+(.+)$",
    "case": r"^CASE\s+(.+)$",
    "default": r"^DEFAULT$",
    "endswitch": r"^ENDSWITCH$",
    "for": r"^FOR\s+[A-Za-z_]\w*\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?$",
    "endfor": r"^ENDFOR$",
    "while": r"^WHILE\s+(.+)$",
    "endwhile": r"^ENDWHILE$",
    "loop": r"^LOOP$",
    "endloop": r"^ENDLOOP$",
    "repeat": r"^REPEAT$",
    "until": r"^UNTIL\s+(.+)$",
    "exit": r"^EXIT$",
    "bas": rf"^BAS\s*\(\s*#(\w+)\s*,\s*({_NUM})\s*\)$",
    "motion": r"^(PTP|LIN)\s*\{([^}]*)\}\s*(C_DIS|C_PTP)?$",
    "vel_cp": r"^\$VEL\.CP\s*=\s*(.+)$",
    "vel_axis": r"^\$VEL_AXIS\s*\[\s*(\d+)\s*\]\s*=\s*(.+)$",
    "acc_axis": r"^\$ACC_AXIS\s*\[\s*(\d+)\s*\]\s*=\s*(.+)$",
    "apo_cdis": r"^\$APO\.CDIS\s*=\s*(.+)$",
    "ov_pro": r"^\$OV_PRO\s*=\s*(.+)$",
    "base": r"^\$BASE\s*=\s*(?:(\$NULLFRAME)|BASE_DATA\s*\[\s*(\d+)\s*\])$",
    "tool": r"^\$TOOL\s*=\s*TOOL_DATA\s*\[\s*(\d+)\s*\]$",
    "ori_type": r"^\$ORI_TYPE\s*=\s*#(\w+)$",
    # writes to the frame tables (whole frame or one field); not modelled
    "frame_data": r"^(TOOL_DATA|BASE_DATA)\s*(?:\[([^\]]*)\])?((?:\s*\.\s*\w+)*)\s*=\s*(.+)$",
    "pdat_member": r"^PDAT_ACT\s*\.\s*(VEL|ACC|APO_DIST)\s*=\s*(.+)$",
    "pdat_whole": r"^PDAT_ACT\s*=\s*(.+)$",
    "bwdstart": r"^\$BWDSTART\s*=\s*(TRUE|FALSE)$",
    "advance": r"^\$ADVANCE\s*=\s*(\d+)$",
    "rsi": r"^(?:[A-Za-z_]\w*\s*=\s*)?RSI_(CREATE|ON|OFF)\s*\((.*)\)$",
    "user_assign": r"^([A-Za-z_]\w*)(?:\s*\[[^\]]*\])?(?:\s*\.\s*\w+)*\s*=\s*(.+)$",
    "sys_assign": r"^\$(\w+)\s*(\[[^\]]*\])?((?:\s*\.\s*\w+)*)\s*=\s*(.+)$",
}.items()}

# KRL operators that may stand before "(" without being a call
_OPERATORS = {"NOT", "AND", "OR", "EXOR", "B_AND", "B_OR", "B_NOT", "B_EXOR", "DIV", "MOD"}
_KEYWORDS = _OPERATORS | {"IF", "THEN", "ELSE", "ENDIF", "FOR", "TO", "STEP", "ENDFOR", "WHILE", "ENDWHILE",
                          "LOOP", "ENDLOOP", "REPEAT", "UNTIL", "SWITCH", "CASE", "DEFAULT", "ENDSWITCH",
                          "EXIT", "GOTO", "HALT", "WAIT", "DECL", "DEF", "END", "TRUE", "FALSE"}


def _pure(expr):
    """True if ``expr`` is a literal, a variable or arithmetic / logic on them, with no call of a
    function or subroutine (an identifier followed by "(") and nothing else the checker cannot read."""
    e = re.sub(r'"[^"]*"', '""', expr)                        # string literals
    if not e.strip() or not re.fullmatch(r"[\w\s$.\[\],+\-*/<>=()#'{}:\"]*", e):
        return False
    for m in re.finditer(r"([A-Za-z_$][\w$]*)\s*\(", e):
        if m.group(1).upper() not in _OPERATORS:
            return False
    return True


class _Unknown:
    """A setting (or the pose) changed by a statement the checker cannot interpret."""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "unknown"


UNKNOWN = _Unknown()
_UNK_WHY = ("changed by a statement not interpreted or invalid, or inherited from the caller at the start "
            "of a DEF")


class Multi(frozenset):
    """The setting may hold any of these values at this motion (it was changed inside a branch or loop)."""

    def __repr__(self):
        return "one of {" + ", ".join(sorted(map(repr, self))) + "}"


def _options(v):
    return set(v) if isinstance(v, Multi) else {v}


# --------------------------------------------------------------------------- parsing
@dataclass
class Statement:
    line: int
    text: str
    kind: str          # motion | setting | frame | no_motion | not_interpreted
    what: str = ""
    op: tuple | None = field(default=None, repr=False)   # effect on the interpreter state


@dataclass
class Motion:
    line: int
    text: str
    kind: str                       # PTP | LIN
    target_type: str                # axis | cartesian
    values: dict                    # A1..A6 (deg) or X Y Z A B C (mm, deg)
    s: int | None = None
    t: int | None = None
    external: dict = field(default_factory=dict)
    approx: str | None = None
    state: dict = field(default_factory=dict)   # settings in force at this motion (Multi: one of several)
    prev: frozenset = frozenset()   # motions that may run just before ("start": program start, UNKNOWN)
    context: tuple = ()             # enclosing IF / SWITCH / loop keywords
    section: int = 0                # index of the DEF it belongs to (Program.sections)


@dataclass
class Issue:
    code: str
    line: int
    message: str

    @property
    def severity(self):
        return "error" if self.code in ERRORS else "flag"

    def as_dict(self):
        return {"code": self.code, "severity": self.severity, "line": self.line, "message": self.message}


@dataclass
class Program:
    statements: list
    motions: list
    issues: list                   # parse / frame errors and statements not interpreted
    sections: list = field(default_factory=list)   # one per DEF: name, line, end_line, first


def _strip_comment(raw):
    out, q = [], False
    for ch in raw:
        if ch == '"':
            q = not q
        if ch == ";" and not q:
            break
        out.append(ch)
    return "".join(out).strip()


def _num(s):
    s = s.strip()
    return float(s) if re.fullmatch(_NUM, s) else None


def _st_value(v):
    v = v.strip()
    m = re.fullmatch(r"'B([01]+)'", v, re.IGNORECASE)
    if m:
        return int(m.group(1), 2)
    if re.fullmatch(r"\d+", v):
        return int(v)
    raise ValueError(f"S / T value {v!r}")


def _parse_aggregate(body):
    body = re.sub(r"^\s*(E6POS|POS|E6AXIS|AXIS|FRAME)\s*:", "", body, flags=re.IGNORECASE)
    vals, ext, s, t = {}, {}, None, None
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"([A-Za-z]\w*)\s+(.+)", item)
        if not m:
            raise ValueError(f"component {item!r}")
        k, v = m.group(1).upper(), m.group(2)
        if k in ("S", "T"):
            if k == "S":
                s = _st_value(v)
            else:
                t = _st_value(v)
            continue
        x = _num(v)
        if x is None:
            raise ValueError(f"{k} is not a number ({v.strip()!r})")
        if re.fullmatch(r"E[1-6]", k):
            ext[k] = x
        elif k in ("X", "Y", "Z", "A", "B", "C") or re.fullmatch(r"A[1-6]", k):
            if k in vals:
                raise ValueError(f"{k} given twice")
            vals[k] = x
        else:
            raise ValueError(f"unknown component {k}")
    axes = [f"A{i}" for i in range(1, 7)]
    cart = ["X", "Y", "Z", "A", "B", "C"]
    if any(k in vals for k in axes):
        if set(vals) != set(axes) or s is not None or t is not None:
            raise ValueError("an axis target needs exactly A1..A6 (E1..E6 optional)")
        return "axis", {k: vals[k] for k in axes}, s, t, ext
    if set(vals) != set(cart):
        raise ValueError("a Cartesian target needs X, Y, Z, A, B, C")
    return "cartesian", {k: vals[k] for k in cart}, s, t, ext


# interpreter state keys (one frozenset of possible values each)
_AXES = range(6)
SETTING_KEYS = (["vel_cp_m_s", "apo_cdis", "ov_pro", "tool", "base", "ori_type", "rsi_on", "pdat_vel", "pdat_acc"]
                + [("vel_axis_pct", i) for i in _AXES] + [("acc_axis_pct", i) for i in _AXES])
# what a statement that is not interpreted may change: every modelled setting and the pose
_ALL_MODELLED = ["vel_cp_m_s", "tool", "base", "ori_type", "pdat_vel", "pdat_acc"] \
    + [("vel_axis_pct", i) for i in _AXES] + [("acc_axis_pct", i) for i in _AXES]


def _unknown_effect(s):
    """("unknown", keys, pose) for a statement that is not interpreted: an assignment to a system
    variable with a call-free right-hand side changes at most that variable; anything else (a call,
    a BAS() function not modelled, unknown syntax) may change every modelled setting and move the robot."""
    m = _RE["sys_assign"].match(s)
    if m and _pure(m.group(4)):
        root, idx, mem = m.group(1).upper(), m.group(2) or "", re.sub(r"\s", "", m.group(3) or "").upper()
        k = re.fullmatch(r"\[\s*(\d+)\s*\]", idx)
        axes = [int(k.group(1)) - 1] if k and 1 <= int(k.group(1)) <= 6 else list(_AXES)
        keys = {"TOOL": ["tool"], "BASE": ["base"], "ORI_TYPE": ["ori_type"],
                "VEL": ["vel_cp_m_s"] if mem in ("", ".CP") else [],
                "VEL_AXIS": [("vel_axis_pct", i) for i in axes],
                "ACC_AXIS": [("acc_axis_pct", i) for i in axes]}.get(root, [])
        return ("unknown", tuple(keys), False)
    return ("unknown", tuple(_ALL_MODELLED), True)


def _valid(v):
    """A speed / acceleration value the checker can use: finite and > 0."""
    return v is not None and math.isfinite(v) and v > 0


def _classify(i, raw, s, issues):
    """One statement: (Statement, control keyword or None)."""
    R = {k: r.match(s) for k, r in _RE.items()}

    def st(kind, what="", op=None):
        return Statement(i, raw.rstrip(), kind, what, op)

    def not_interp(msg=None, op=None):
        issues.append(Issue("not_interpreted", i, msg or f"not interpreted: {s}"))
        return st("not_interpreted", "", op or _unknown_effect(s))

    def invalid(name, v, keys):
        issues.append(Issue("invalid_setting", i, f"{name} = {v:g}: not a positive finite value (the setting "
                                                  f"is unknown from here on): {s}"))
        return st("invalid_setting", f"{name} invalid", ("unknown", tuple(keys), False))

    def frame(which, n):
        if n != (MODELLED_TOOL if which == "TOOL" else MODELLED_BASE):
            issues.append(Issue("frame_not_modelled", i, f"{which} {n}: frame not modelled (only "
                                f"{which} {MODELLED_TOOL if which == 'TOOL' else MODELLED_BASE} of cell.yaml)"))
        return st("frame", f"{which} {n}", ("set", which.lower(), n))

    # control flow: the structure is kept even when the condition is not interpreted
    for kw in ("if", "else", "endif", "switch", "case", "default", "endswitch", "for", "endfor",
               "while", "endwhile", "loop", "endloop", "repeat", "until", "exit"):
        if R[kw]:
            exprs = [g for g in R[kw].groups() if g is not None]
            if all(_pure(e) for e in exprs):
                return st("no_motion", kw.upper()), kw
            return not_interp(f"condition not interpreted (may call a function): {s}",
                              ("unknown", tuple(_ALL_MODELLED), True)), kw
    if R["attribute"]:
        return st("no_motion", "editor attribute"), None
    if R["def"]:
        return st("no_motion", "DEF", ("def", R["def"].group(1))), None
    if R["end"]:
        return st("no_motion", "END"), None
    if R["halt"]:
        return st("no_motion", "HALT"), None
    if R["wait"]:
        return st("no_motion", "WAIT SEC"), None
    if R["decl"]:
        init = R["decl"].group(1)
        if "=" in init and not _pure(init.split("=", 1)[1]):
            return not_interp(), None
        return st("no_motion", "DECL"), None
    if R["rsi"]:
        w = R["rsi"].group(1).upper()
        if R["rsi"].group(2).strip() and not _pure(R["rsi"].group(2)):
            return not_interp(), None
        op = ("set", "rsi_on", True) if w == "ON" else ("set", "rsi_on", False) if w == "OFF" else None
        return st("no_motion", f"RSI_{w}", op), None
    if R["bas"]:
        what, n = R["bas"].group(1).upper(), float(R["bas"].group(2))
        if what in ("TOOL", "BASE"):
            return frame(what, int(n)), None
        if what == "INITMOV":
            return st("setting", "BAS(#INITMOV): controller defaults (not modelled: speeds unset)",
                      ("initmov",)), None
        if what in ("VEL_PTP", "PTP_PARAMS") and not _valid(n):
            return invalid(f"BAS(#{what})", n, [(k, i) for k in ("vel_axis_pct", "acc_axis_pct") for i in _AXES]
                           if what == "PTP_PARAMS" else [("vel_axis_pct", i) for i in _AXES]), None
        if what == "VEL_PTP":
            return st("setting", f"$VEL_AXIS[1..6] = {n:g} %", ("axes", "vel_axis_pct", tuple(_AXES), n)), None
        if what == "PTP_PARAMS":
            return st("setting", f"BAS(#PTP_PARAMS, {n:g}): $VEL_AXIS[1..6] <= max({n:g}, PDAT_ACT.VEL) %, "
                                 "$ACC_AXIS[1..6] = PDAT_ACT.ACC %", ("ptp_params", n)), None
        return not_interp(), None
    if R["motion"]:
        kind, body, approx = R["motion"].group(1).upper(), R["motion"].group(2), R["motion"].group(3)
        try:
            tt, vals, S, T, ext = _parse_aggregate(body)
            if kind == "LIN" and tt == "axis":
                raise ValueError("LIN to an axis target")
        except ValueError as e:
            issues.append(Issue("parse_error", i, f"motion not interpreted ({e}): {s}"))
            return st("not_interpreted", "motion", ("unknown", (), True)), None
        m = Motion(i, s, kind, tt, vals, S, T, ext, approx.upper() if approx else None)
        return st("motion", kind, ("motion", m)), None
    if R["vel_cp"]:
        v = _num(R["vel_cp"].group(1))
        if v is None:
            return not_interp(f"$VEL.CP is not a number (speed unknown): {s}",
                              ("unknown", ("vel_cp_m_s",), not _pure(R["vel_cp"].group(1)))), None
        if not _valid(v):
            return invalid("$VEL.CP", v, ("vel_cp_m_s",)), None
        return st("setting", f"$VEL.CP = {v:g} m/s", ("set", "vel_cp_m_s", v)), None
    if R["vel_axis"] or R["acc_axis"]:
        key = "vel_axis_pct" if R["vel_axis"] else "acc_axis_pct"
        m = R["vel_axis"] or R["acc_axis"]
        k, v = int(m.group(1)), _num(m.group(2))
        if not 1 <= k <= 6 or v is None:
            return not_interp(), None
        name = "$VEL_AXIS" if key == "vel_axis_pct" else "$ACC_AXIS"
        if not _valid(v):
            return invalid(f"{name}[{k}]", v, ((key, k - 1),)), None
        return st("setting", f"{name}[{k}] = {v:g} %", ("axes", key, (k - 1,), v)), None
    if R["apo_cdis"]:
        if not _pure(R["apo_cdis"].group(1)):
            return not_interp(), None
        return st("setting", "$APO.CDIS (blending not modelled)"), None
    if R["ov_pro"]:
        if not _pure(R["ov_pro"].group(1)):
            return not_interp(), None
        return st("no_motion", "$OV_PRO (speeds are checked at 100 % override)"), None
    if R["base"]:
        return frame("BASE", MODELLED_BASE if R["base"].group(1) else int(R["base"].group(2))), None
    if R["tool"]:
        return frame("TOOL", int(R["tool"].group(1))), None
    if R["ori_type"]:
        v = "#" + R["ori_type"].group(1).upper()
        if v not in ("#CONSTANT", "#VAR"):
            return not_interp(f"$ORI_TYPE {v} not modelled (only #VAR and #CONSTANT): {s}",
                              ("unknown", ("ori_type",), False)), None
        return st("setting", f"$ORI_TYPE = {v}", ("set", "ori_type", v)), None
    if R["pdat_member"]:
        mem, rhs = R["pdat_member"].group(1).upper(), R["pdat_member"].group(2)
        v = _num(rhs)
        if mem == "APO_DIST":
            if not _pure(rhs):
                return not_interp(), None
            return st("setting", "PDAT_ACT.APO_DIST (blending, no effect on the checks)"), None
        key = "pdat_vel" if mem == "VEL" else "pdat_acc"
        if v is None:
            return not_interp(f"PDAT_ACT.{mem} is not a number (value unknown): {s}",
                              ("unknown", (key,), not _pure(rhs))), None
        if not _valid(v):
            return invalid(f"PDAT_ACT.{mem}", v, (key,)), None
        return st("setting", f"PDAT_ACT.{mem} = {v:g} % (used by BAS(#PTP_PARAMS))", ("set", key, v)), None
    if R["pdat_whole"]:
        rhs = R["pdat_whole"].group(1).strip()
        if not _pure(rhs):
            return not_interp(), None
        vals = {}
        lit = re.fullmatch(r"\{(.*)\}", rhs)
        if lit:
            for item in lit.group(1).split(","):
                mm = re.fullmatch(rf"\s*(\w+)\s+({_NUM})\s*", item)
                if mm:
                    vals[mm.group(1).upper()] = float(mm.group(2))
        vel, acc = vals.get("VEL", UNKNOWN), vals.get("ACC", UNKNOWN)
        return st("setting", "PDAT_ACT = " + ("data set (VEL / ACC unknown here)" if not lit else rhs),
                  ("pdat", vel, acc)), None
    if R["bwdstart"]:
        return st("setting", "$BWDSTART (no effect on the checks)"), None
    if R["advance"]:
        return st("setting", "$ADVANCE (no effect on the checks)"), None
    if R["frame_data"]:
        table, idx = R["frame_data"].group(1).upper(), (R["frame_data"].group(2) or "").strip()
        which = "tool" if table == "TOOL_DATA" else "base"
        n = int(idx) if re.fullmatch(r"\d+", idx) else "*"
        frame_op = ("frame_write", which, n, not _pure(R["frame_data"].group(4)))
        return not_interp(f"write to {table}" + (f"[{n}]" if n != "*" else "") + " not modelled: "
                          + ("that frame" if n != "*" else f"every {table} frame") + " is unknown from here on: "
                          + s, frame_op), None
    if R["user_assign"] and R["user_assign"].group(1).upper() not in _KEYWORDS:
        if _pure(R["user_assign"].group(2)):
            return st("no_motion", f"assignment to user variable {R['user_assign'].group(1)}"), None
        return not_interp(f"not interpreted (right-hand side may call a function): {s}"), None
    first = re.match(r"[A-Za-z_$]\w*", s)
    if first and first.group(0).upper() in MOTION_WORDS:
        issues.append(Issue("parse_error", i, f"motion not interpreted: {s}"))
        return st("not_interpreted", "motion", ("unknown", (), True)), None
    return not_interp(), None


_OPENERS = {"if": ("endif",), "switch": ("endswitch",), "for": ("endfor",), "while": ("endwhile",),
            "loop": ("endloop",), "repeat": ("until",)}


def _build_tree(items, issues):
    """Nest the statements: ("op", Statement) | ("branch", kw, [blocks], exhaustive) | ("loop", kw, block)."""
    root = []
    stack = []        # (keyword, statement, blocks, node list being filled)
    cur = root
    for stmt, kw in items:
        if kw is None:
            cur.append(("op", stmt))
            continue
        if kw == "exit":
            if not any(e[0] in ("for", "while", "loop", "repeat") for e in stack):
                issues.append(Issue("parse_error", stmt.line, "EXIT outside a loop"))
            cur.append(("exit", stmt))
            continue
        if stmt.op is not None:          # a control statement that is not interpreted
            cur.append(("op", stmt))
        if kw in _OPENERS:
            blocks = [[]]
            stack.append((kw, stmt, blocks))
            cur = blocks[0]
        elif kw in ("else", "case", "default"):
            want = "if" if kw == "else" else "switch"
            if not stack or stack[-1][0] != want:
                issues.append(Issue("parse_error", stmt.line, f"{kw.upper()} without {want.upper()}"))
                continue
            blocks = stack[-1][2]
            if kw in ("case", "default") and blocks == [[]] and not blocks[0]:
                blocks.clear()
            blocks.append([])
            if kw in ("else", "default"):
                stack[-1] = (*stack[-1][:2], blocks, True)
            cur = blocks[-1]
        else:   # closer
            if not stack or kw not in _OPENERS[stack[-1][0]]:
                issues.append(Issue("parse_error", stmt.line, f"{kw.upper()} without a matching opener"))
                continue
            entry = stack.pop()
            okw, _, blocks = entry[:3]
            exhaustive = len(entry) > 3
            node = ("loop", okw, blocks[0]) if okw in ("for", "while", "loop", "repeat") \
                else ("branch", okw, blocks, exhaustive)
            cur = stack[-1][2][-1] if stack else root
            cur.append(node)
    for kw, stmt, *_ in stack:
        issues.append(Issue("parse_error", stmt.line, f"{kw.upper()} without {_OPENERS[kw][0].upper()}"))
    return root


def _union(states):
    out = {}
    for S in states:
        for k, v in S.items():
            out[k] = out.get(k, frozenset()) | v
    return out


class _Interpreter:
    """Abstract interpretation of the statement tree: each setting holds the set of values it may have.
    IF / SWITCH: every branch is followed and the results are merged (a branch that may be skipped
    keeps the incoming values). Loops: the body is run until the set of reachable states no longer
    grows; every state a loop can reach (also at EXIT) flows on after it."""

    def __init__(self, motions, frame_writes=frozenset()):
        self.motions = motions
        # every TOOL_DATA / BASE_DATA write in the file ({("tool" | "base", n | "*")}): a call may run any of them
        self.frame_writes = frame_writes
        self.rec = {}            # id(motion) -> merged state
        self.ctx = {}
        self.exits = []

    def apply(self, stmt, S):
        op = stmt.op
        if op is None:
            return S
        S = dict(S)
        t = op[0]
        if t == "set":
            S[op[1]] = frozenset([op[2]])
            if op[1] in ("tool", "base") and op[2] != 0 and S["dirty_" + op[1]] & {op[2], "*"}:
                # TOOL_DATA[n] / BASE_DATA[n] may have been written by a statement not modelled
                S[op[1]] = frozenset([UNKNOWN])
        elif t == "frame_write":
            which, n, call = op[1], op[2], op[3]
            S["dirty_" + which] = S["dirty_" + which] | {n}
            # conservative: the active frame is unknown at once if it may be the one written
            if n == "*" or n in S[which]:
                S[which] = frozenset([UNKNOWN])
            if call:
                S = self.apply(Statement(0, "", "not_interpreted", "", ("unknown", tuple(_ALL_MODELLED), True)), S)
        elif t == "axes":
            for i in op[2]:
                S[(op[1], i)] = frozenset([op[3]])
        elif t == "initmov":
            S["vel_cp_m_s"] = frozenset([None])
            for i in _AXES:
                S[("vel_axis_pct", i)] = S[("acc_axis_pct", i)] = frozenset([None])
        elif t == "ptp_params":
            n = op[1]
            vel = frozenset(UNKNOWN if v is UNKNOWN else max(n, 100.0 if v is None else v)
                            for v in S["pdat_vel"])
            acc = frozenset(v for v in S["pdat_acc"])
            for i in _AXES:
                S[("vel_axis_pct", i)] = vel
                S[("acc_axis_pct", i)] = acc
        elif t == "pdat":
            S["pdat_vel"], S["pdat_acc"] = frozenset([op[1]]), frozenset([op[2]])
        elif t == "unknown":
            for k in op[1]:
                S[k] = frozenset([UNKNOWN])
            if op[2]:
                S["prev"] = frozenset([UNKNOWN])
                for which, n in self.frame_writes:      # a call may run any frame write of the file
                    S["dirty_" + which] = S["dirty_" + which] | {n}
        elif t == "motion":
            m = op[1]
            key = id(m)
            self.rec[key] = _union([self.rec[key], S]) if key in self.rec else dict(S)
            S["prev"] = frozenset([key])
        return S

    def run(self, nodes, S, ctx=()):
        for n in nodes:
            if n[0] == "op":
                if n[1].op is not None and n[1].op[0] == "motion":
                    self.ctx[id(n[1].op[1])] = ctx
                S = self.apply(n[1], S)
            elif n[0] == "exit":
                if self.exits:
                    self.exits[-1].append(S)
            elif n[0] == "branch":
                outs = [self.run(b, S, ctx + (n[1].upper(),)) for b in n[2]]
                if not n[3]:
                    outs.append(S)
                S = _union(outs)
            else:   # loop
                head = S
                self.exits.append([])
                for _ in range(1000):
                    end = self.run(n[2], head, ctx + (n[1].upper(),))
                    new = _union([head, end])
                    if new == head:
                        break
                    head = new
                S = _union([head] + self.exits.pop())
        return S


def _collapse(v):
    return next(iter(v)) if len(v) == 1 else Multi(v)


def _initial_state(first, frame_writes=frozenset()):
    """Settings at the start of a DEF. The first DEF of the file (the program): nothing set (None), pose from
    --start-axis. Any later DEF (a subprogram / function called from elsewhere): every setting, the pose and
    the frame tables may come from the caller, so all are unknown."""
    if first:
        S = {k: frozenset([False if k == "rsi_on" else None]) for k in SETTING_KEYS}
        S["prev"] = frozenset(["start"])
        S["dirty_tool"] = S["dirty_base"] = frozenset()
        return S
    S = {k: frozenset([False, True] if k == "rsi_on" else [UNKNOWN]) for k in SETTING_KEYS}
    S["prev"] = frozenset(["def_start"])
    S["dirty_tool"] = frozenset(n for w, n in frame_writes if w == "tool")
    S["dirty_base"] = frozenset(n for w, n in frame_writes if w == "base")
    return S


def parse_program(text):
    """Read a .src text into statements, motions (with the settings that may be in force) and issues.
    Each DEF is its own section: at every DEF after the first, the pose and every setting are reset to unknown."""
    issues, items = [], []
    for i, raw in enumerate(text.splitlines(), start=1):
        s = _strip_comment(raw)
        if s:
            items.append(_classify(i, raw, s, issues))
    statements = [st for st, _ in items]
    motions = [st.op[1] for st in statements if st.op is not None and st.op[0] == "motion"]
    frame_writes = frozenset((st.op[1], st.op[2]) for st in statements
                             if st.op is not None and st.op[0] == "frame_write")
    # split into DEF sections (statements before the first DEF belong to it)
    parts, seen_def = [[]], False
    for item in items:
        op = item[0].op
        if op is not None and op[0] == "def":
            if seen_def:
                parts.append([])
            seen_def = True
        parts[-1].append(item)
    sections = []
    it = _Interpreter(motions, frame_writes)
    S0 = None
    for k, part in enumerate(parts):
        d = next((st for st, _ in part if st.op is not None and st.op[0] == "def"), None)
        name = d.op[1] if d is not None else "(no DEF)"
        sec = {"name": name, "line": d.line if d is not None else part[0][0].line,
               "end_line": part[-1][0].line, "first": k == 0}
        sections.append(sec)
        for st, _ in part:
            if st.op is not None and st.op[0] == "motion":
                st.op[1].section = k
        tree = _build_tree(part, issues)
        S = _initial_state(k == 0, frame_writes)
        if k == 0:
            S0 = S
        it.run(tree, S)
    index = {id(m): k for k, m in enumerate(motions)}
    for m in motions:
        S = it.rec.get(id(m))
        if S is None:          # inside a structure that never runs (cannot happen with the rules above)
            S = S0
        m.state = {k: _collapse(S[k]) for k in SETTING_KEYS if isinstance(k, str)}
        m.state["vel_axis_pct"] = [_collapse(S[("vel_axis_pct", i)]) for i in _AXES]
        m.state["acc_axis_pct"] = [_collapse(S[("acc_axis_pct", i)]) for i in _AXES]
        m.prev = frozenset(index[p] if isinstance(p, int) and p in index else p for p in S["prev"])
        m.context = it.ctx.get(id(m), ())
    issues.sort(key=lambda x: x.line)
    return Program(statements, motions, issues, sections)


# --------------------------------------------------------------------------- keep-out
def load_keep_out(path):
    """Boxes from a keep-out file: a list, or a mapping with key ``keep_out`` (same entry
    format as ``keep_out`` in cell.yaml)."""
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if d is None:
        return []
    if isinstance(d, dict):
        d = d.get("keep_out") or []
    if not isinstance(d, list):
        raise ValueError(f"{Path(path).name}: expected a list of keep-out boxes")
    return d


# --------------------------------------------------------------------------- checker
class SrcChecker:
    def __init__(self, cfg=None, keep_out=None, keep_out_source="none", start_axis_deg=None):
        """keep_out: extra boxes (list of cell.yaml-style entries) added to cell.yaml keep_out."""
        base = cfg or load_config()
        extra = list(keep_out or [])
        self.cfg = load_config(cell_overrides={"keep_out": list(base.cell.get("keep_out") or []) + extra},
                               planner_overrides=base.overrides.get("planner")) if extra else base
        self.n_keep_out_extra = len(extra)
        self.n_keep_out_cell = len(base.cell.get("keep_out") or [])
        self.keep_out_source = keep_out_source
        self.keep_out_names = {b.name for b in keep_out_boxes(self.cfg)}
        self.p = self.cfg.planner["src_check"]
        self.robot = Robot.from_config(self.cfg)
        self.coll = CollisionModel(self.robot, self.cfg)
        self.tpl = TransitPlanner.from_config(self.robot, self.cfg, self.coll)
        self.lim = self.tpl.lim                                   # URDF limits minus use_margin_deg
        self.start_q = None if start_axis_deg is None else np.radians(np.asarray(start_axis_deg, float))
        bed = self.cfg.bed()
        self.bed = bed
        self.bed_top = float(self.p.get("bed_top_max_z", bed["top_z"]))

    # ---------------------------------------------------------------- helpers
    def _in_limits(self, q):
        return bool(np.all(q >= self.lim[:, 0] - 1e-9) and np.all(q <= self.lim[:, 1] + 1e-9))

    def _branch(self, q):
        return {"S": int(self.robot.status_bits(q)), "T": int(self.robot.turn_bits(q))}

    def _pose(self, m):
        v = m.values
        return xyzabc_to_T([v["X"], v["Y"], v["Z"], v["A"], v["B"], v["C"]])

    def _choose(self, m, T, q_prev, notes):
        """IK solution for a Cartesian target: S / T if given, else the current status
        (nearest), else the solution closest to assumed_ref_axis_deg. None if none."""
        Q, ok = self.robot.ik(T, limits=self.lim)
        Q, ok = Q[0], ok[0]
        k = np.nonzero(ok)[0]
        if not len(k):
            return None, "no IK solution inside the joint limits minus the margin"
        S = self.robot.status_bits(Q[k])
        Tb = self.robot.turn_bits(Q[k])
        if m.s is not None or m.t is not None:
            sel = np.ones(len(k), bool)
            if m.s is not None:
                sel &= (S & 7) == (m.s & 7)
            if m.t is not None:
                sel &= (Tb & 63) == (m.t & 63)
            if not sel.any():
                return None, (f"no IK solution with S {m.s} / T {m.t} inside the limits "
                              f"(solutions exist with S/T {sorted(set(zip(S.tolist(), Tb.tolist())))})")
            kk = k[sel]
            if q_prev is not None:
                kk = kk[np.argsort(np.abs(Q[kk] - q_prev).max(1), kind="stable")]
            if m.kind == "LIN" and q_prev is not None:
                notes.append("S / T on a LIN target are ignored by the controller")
            return Q[kk[0]], f"S {m.s} / T {m.t} from the program"
        if q_prev is not None:
            s0 = int(self.robot.status_bits(q_prev))
            same = k[S == s0]
            if not len(same):
                return None, f"no IK solution with the current status S {s0} inside the limits"
            return Q[same[np.argmin(np.abs(Q[same] - q_prev).max(1))]], "current status, nearest turn"
        ref = np.radians(np.asarray(self.p["assumed_ref_axis_deg"], float))
        j = k[np.argmin(np.abs(Q[k] - ref).max(1))]
        why = ("ASSUMED: no S / T and no preceding axis PTP; IK solution closest to A1..A6 = "
               + ", ".join(f"{a:g}" for a in self.p["assumed_ref_axis_deg"]))
        return Q[j], why

    def _lin_samples(self, T0, T1, ori_type):
        p0, p1 = T0[:3, 3], T1[:3, 3]
        R0 = T0[:3, :3]
        R1 = R0 if ori_type == "#CONSTANT" else T1[:3, :3]
        ang = math.degrees(float(rotation_angle_between(R0, R1)))
        n = int(np.ceil(max(np.linalg.norm(p1 - p0) / self.p["lin_sample_mm"],
                            ang / self.p["lin_sample_deg"]))) + 1
        s = np.linspace(0.0, 1.0, max(n, 2))
        P = p0 + s[:, None] * (p1 - p0)
        if ang > 1e-9:
            R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))(s).as_matrix()
        else:
            R = np.broadcast_to(R0, (len(s), 3, 3))
        return make_transform(R, P)

    def _follow_lin(self, q0, Ts):
        """Follow one IK branch from q0 along the sampled poses. Returns (q samples, problem or None)."""
        Q, ok = self.robot.ik(Ts[1:], limits=self.lim)
        thr = np.radians(self.p["lin_max_joint_step_deg"])
        s0 = int(self.robot.status_bits(q0))
        qs = [q0]
        prob = None
        n = len(Ts) - 1
        for i in range(n):
            if not ok[i].any():
                return np.array(qs), ("reach", f"no IK solution inside the limits at {100 * (i + 1) / n:.0f} % "
                                               "of the LIN")
            d = np.abs(Q[i] - qs[-1]).max(-1)
            d[~ok[i]] = np.inf
            k = int(np.argmin(d))
            qn = Q[i, k]
            if prob is None:
                sn = int(self.robot.status_bits(qn))
                if sn != s0:
                    prob = ("lin_branch", f"configuration change on a LIN (status S {s0} -> {sn}) at "
                                          f"{100 * (i + 1) / n:.0f} %")
                elif d[k] > thr:
                    j = int(np.argmax(np.abs(qn - qs[-1])))
                    prob = ("lin_branch", f"joint step {math.degrees(d[k]):.1f} deg on A{j + 1} > "
                                          f"{self.p['lin_max_joint_step_deg']:g} deg at {100 * (i + 1) / n:.0f} % "
                                          "(the branch followed from the start ends)")
            qs.append(qn)
        return np.array(qs), prob

    # ---------------------------------------------------------------- per motion
    def _evaluate(self, rec, m, Q, exempt_tip):
        """Singularity, clearance, nozzle height and bed edge over the joint samples Q."""
        p = self.p
        flags = rec["flags"]
        a5 = float(np.degrees(np.abs(Q[:, 4])).min())
        elb = float(np.degrees(np.abs(Q[:, 2] - self.robot.delta3)).min())
        wc = self.robot.wrist_center(self.robot.fk_flange(Q))
        sh = float(np.hypot(wc[:, 0], wc[:, 1]).min())
        rec.update(min_abs_a5_deg=round(a5, 2), min_elbow_deg=round(elb, 2), min_shoulder_mm=round(sh, 1))
        sing = []
        if a5 < p["min_abs_a5_deg"]:
            sing.append(f"|A5| {a5:.1f} deg < {p['min_abs_a5_deg']:g}")
        if elb < p["min_elbow_deg"]:
            sing.append(f"elbow {elb:.1f} deg < {p['min_elbow_deg']:g}")
        if sh < p["min_shoulder_mm"]:
            sing.append(f"shoulder {sh:.0f} mm < {p['min_shoulder_mm']:g}")
        if sing:
            flags.append({"code": "singularity", "message": "; ".join(sing)})
        mar = self.coll.margins(Q, exempt_tip=exempt_tip)
        best = (np.inf, None, None)
        for ob, M in mar.items():
            j = np.unravel_index(int(np.argmin(M)), M.shape)
            if M[j] < best[0]:
                best = (float(M[j]), self.coll.names[j[1]], ob)
        c, prim, ob = best
        obl = f"keep-out {ob}" if ob in self.keep_out_names else ob
        rec.update(min_clearance_mm=round(c, 2), clearance_pair=[prim, obl])
        if c < p["min_clearance_mm"]:
            flags.append({"code": "clearance", "message": f"clearance {c:.1f} mm < {p['min_clearance_mm']:g} mm "
                                                          f"({prim} / {obl})"})
        tip = self.robot.fk(Q)[:, :3, 3]
        h = float(tip[:, 2].min() - self.bed_top)
        rec["min_tip_height_mm"] = round(h, 2)
        if h < p["min_nozzle_bed_mm"]:
            flags.append({"code": "nozzle_height", "message": f"tip {h:.2f} mm above the highest bed top "
                                                              f"(Z {self.bed_top:g}) < {p['min_nozzle_bed_mm']:g} mm"})
        if m.kind == "LIN":
            bx, by = self.bed["x"], self.bed["y"]
            d = np.minimum.reduce([tip[:, 0] - bx[0], bx[1] - tip[:, 0], tip[:, 1] - by[0], by[1] - tip[:, 1]])
            dmin = float(d.min())
            rec["min_tip_to_bed_edge_mm"] = round(dmin, 1)
            if dmin < p["bed_edge_margin_mm"]:
                flags.append({"code": "bed_edge", "message": f"tip {dmin:.1f} mm inside the bed outline "
                                                             f"(< {p['bed_edge_margin_mm']:g} mm margin)"})

    def _worst_pct(self, vals, unset, name):
        """Highest possible value per axis (None = not set -> ``unset``); (array, reason) where reason
        names an unknown axis (setting changed by a statement not interpreted) or is None."""
        out, unk, multi = [], [], []
        for i, v in enumerate(vals):
            opts = _options(v)
            if UNKNOWN in opts:
                unk.append(f"{name}[{i + 1}]")
                out.append(np.nan)
                continue
            if len(opts) > 1:
                multi.append(i + 1)
            out.append(max(unset if x is None else float(x) for x in opts))
        return np.array(out), (", ".join(unk) if unk else None), multi

    def _starts(self, m, ends, idx):
        """Possible joint values before motion ``idx``: (list of q, reason why the start is unknown or None)."""
        cands = []
        for p in m.prev:
            if p is UNKNOWN:
                return [], "a statement before it that is not interpreted may have moved the robot"
            if p == "def_start":
                return [], ("first motion of a DEF after the first one: its start pose and settings come "
                            "from the caller")
            if p == "start":
                if self.start_q is None:
                    return [], "first motion of the program (give --start-axis)"
                cands.append(self.start_q.copy())
                continue
            if p >= idx or p not in ends:
                return [], "the motion before it is later in a loop body (start depends on the loop)"
            for q in ends[p]:
                if q is None:
                    return [], "the end pose of the motion before it is unknown"
                cands.append(q)
        out = []
        for q in cands:
            if not any(np.allclose(q, r, atol=1e-9) for r in out):
                out.append(q)
        return out, None

    def check_program(self, prog: Program):
        p = self.p
        ends = {}
        out = []
        for idx, m in enumerate(prog.motions):
            st = m.state
            rec = {"line": m.line, "def": prog.sections[m.section]["name"] if prog.sections else None,
                   "kind": m.kind, "target_type": m.target_type,
                   "target": {k: v for k, v in m.values.items()}, "approx": m.approx,
                   "flags": [], "notes": [], "not_checked": []}
            out.append(rec)
            if m.s is not None:
                rec["target"]["S"] = m.s
            if m.t is not None:
                rec["target"]["T"] = m.t
            if m.external:
                rec["notes"].append("external axes " + ", ".join(sorted(m.external)) + " not modelled")
            if m.approx:
                rec["notes"].append(f"{m.approx} blending not modelled (checked through the programmed point)")
            if m.context:
                rec["notes"].append("inside " + " / ".join(m.context) + ": checked as if executed, with every "
                                    "setting state the branches / loop can reach (worst case)")
            if True in _options(st["rsi_on"]):
                rec["notes"].append("RSI on: a PC can set the override (1..100 %) and RSI corrections are "
                                    "not modelled")
            starts, unknown_why = self._starts(m, ends, idx)
            if len(starts) > 1:
                rec["notes"].append(f"start pose depends on a branch / loop: checked from each of the "
                                    f"{len(starts)} possible start poses")
            rec["start_known"] = unknown_why is None
            rec["speed_mm_s"] = None
            # frames: a Cartesian motion needs TOOL 3 and BASE 0 on every path to it
            if m.target_type == "cartesian":
                tool, base = _options(st["tool"]), _options(st["base"])
                missing = " and ".join(n for n, o in (("TOOL", tool), ("BASE", base)) if None in o)
                if missing:
                    prog.issues.append(Issue("frame_not_modelled", m.line,
                                             f"{missing} not set before this Cartesian motion"))
                bad = [f"${n} unknown ({_UNK_WHY})" for n, o in
                       (("TOOL", tool), ("BASE", base)) if UNKNOWN in o]
                if not bad and (tool != {MODELLED_TOOL} or base != {MODELLED_BASE}):
                    bad = ["frame not modelled" + (" (depends on a branch: TOOL " + repr(st["tool"]) + ", BASE "
                                                   + repr(st["base"]) + ")" if len(tool) > 1 or len(base) > 1 else "")]
                if bad:
                    rec["not_checked"] += [f"not checked: {b}" for b in bad]
                    if m.kind == "LIN" and UNKNOWN in _options(st["vel_cp_m_s"]):
                        rec["not_checked"].append(f"not checked: $VEL.CP unknown ({_UNK_WHY})")
                    if unknown_why is not None:
                        rec["not_checked"].append(f"start unknown: {unknown_why}")
                    rec["checked"] = False
                    ends[idx] = [None]
                    continue
            ends[idx] = []
            recs = []
            for q0 in (starts if unknown_why is None else [None]):
                r = {"flags": [], "notes": [], "not_checked": []}
                q1 = self._check_motion(m, q0, r, unknown_why)
                recs.append(r)
                ends[idx].append(q1)
            self._merge(rec, recs)
            rec["checked"] = not rec["not_checked"]
        return out

    @staticmethod
    def _merge(rec, recs):
        for r in recs:
            for key in ("flags", "notes", "not_checked"):
                for x in r[key]:
                    if x not in rec[key]:
                        rec[key].append(x)
        first = recs[0]
        for k, v in first.items():
            if k not in ("flags", "notes", "not_checked"):
                rec[k] = v
        if len(recs) > 1:
            for k in ("speed_mm_s", "speed_peak_ramp_mm_s"):
                vals = [r[k] for r in recs if r.get(k) is not None]
                if vals:
                    rec[k] = max(vals)
            for k in ("min_abs_a5_deg", "min_elbow_deg", "min_shoulder_mm", "min_tip_height_mm",
                      "min_tip_to_bed_edge_mm"):
                vals = [r[k] for r in recs if k in r]
                if vals:
                    rec[k] = min(vals)
            cl = [r for r in recs if "min_clearance_mm" in r]
            if cl:
                w = min(cl, key=lambda r: r["min_clearance_mm"])
                rec["min_clearance_mm"], rec["clearance_pair"] = w["min_clearance_mm"], w["clearance_pair"]
            rec["n_samples"] = sum(r.get("n_samples", 0) for r in recs)
            rec["n_starts"] = len(recs)

    def _check_motion(self, m, q_prev, rec, unknown_why):
        """Check motion m from joint values q_prev (None: unknown). Fills rec, returns the end joints or None."""
        p = self.p
        st = m.state
        nc = rec["not_checked"]
        if q_prev is None:
            nc.append(f"start unknown, not checked (path{' and speed' if m.kind == 'PTP' else ''}; "
                      f"target pose checked): {unknown_why}")
        # speed settings in force (worst case over every value they may hold)
        if m.kind == "LIN":
            opts = _options(st["vel_cp_m_s"])
            rec["speed_source"] = "$VEL.CP at 100 % override"
            if UNKNOWN in opts:
                nc.append(f"not checked: $VEL.CP unknown ({_UNK_WHY})")
            elif None in opts:
                rec["flags"].append({"code": "speed", "message": "$VEL.CP not set by the program on every path "
                                                                 "to this motion (controller default unknown)"})
            else:
                v = max(opts)
                rec["speed_mm_s"] = round(v * 1000.0, 3)
                if len(opts) > 1:
                    rec["notes"].append("$VEL.CP depends on a branch / loop (" + ", ".join(
                        f"{x * 1000:g}" for x in sorted(opts)) + " mm/s): checked at the highest")
                if v * 1000.0 > p["max_tcp_speed_mm_s"]:
                    rec["flags"].append({"code": "speed", "message": f"$VEL.CP {v * 1000:.0f} mm/s > "
                                                                     f"{p['max_tcp_speed_mm_s']:g} mm/s"})
        vel, vel_unk, vel_multi = self._worst_pct(st["vel_axis_pct"], p["unset_vel_axis_pct"], "$VEL_AXIS")
        acc, acc_unk, acc_multi = self._worst_pct(st["acc_axis_pct"], p["unset_acc_axis_pct"], "$ACC_AXIS")
        # target
        if m.target_type == "axis":
            q1 = np.radians([m.values[f"A{i}"] for i in range(1, 7)])
            why = "axis target"
            if not self._in_limits(q1):
                bad = [f"A{i + 1}" for i in range(6) if not self.lim[i, 0] - 1e-9 <= q1[i] <= self.lim[i, 1] + 1e-9]
                rec["flags"].append({"code": "reach", "message": f"{', '.join(bad)} outside the joint limits "
                                                                 "minus the margin"})
        else:
            T1 = self._pose(m)
            q1, why = self._choose(m, T1, q_prev, rec["notes"])
            if q1 is None:
                rec["flags"].append({"code": "reach", "message": f"target not reachable: {why}"})
                rec["branch"] = None
                if m.kind == "PTP":
                    rec["speed_source"] = "not checked (target not reachable)"
                return None
        # path
        if m.kind == "PTP":
            if vel_unk:
                nc.append(f"not checked (speed): {vel_unk} unknown ({_UNK_WHY})")
            if q_prev is None:
                Q = q1[None]
                rec["speed_source"] = "start unknown, not checked (path and speed); target pose checked"
            elif vel_unk:
                Q = self.tpl.ptp_samples(q_prev, q1, p["ptp_sample_deg"])
                rec["speed_source"] = "not checked: axis speed unknown"
            else:
                Q = self.tpl.ptp_samples(q_prev, q1, p["ptp_sample_deg"])
                peak, cruise, _ = self.tpl.ptp_speed(Q, vel, None if acc_unk else acc)
                rec["speed_mm_s"] = round(cruise, 1)
                rec["speed_peak_ramp_mm_s"] = None if acc_unk else round(peak, 1)
                rec["speed_source"] = (
                    "PTP cruise bound (flag value): TCP speed of the synchronous PTP with the leading axis at its "
                    "$VEL_AXIS speed (URDF axis speeds, unverified), independent of the axis accelerations; "
                    "$VEL_AXIS " + "/".join(f"{x:g}" for x in vel) + " %. Ramp-limited peak "
                    + ("not computed ($ACC_AXIS unknown)" if acc_unk else
                       f"{peak:.1f} mm/s at $ACC_AXIS " + "/".join(f"{x:g}" for x in acc)
                       + " % of the ASSUMED axis accelerations (information only)"))
                if vel_multi or acc_multi:
                    rec["notes"].append("$VEL_AXIS / $ACC_AXIS depend on a branch / loop: checked at the highest "
                                        "value on every axis")
                if cruise > p["max_tcp_speed_mm_s"]:
                    extra = "" if acc_unk else f"; ramp-limited peak {peak:.0f} mm/s with assumed accelerations"
                    rec["flags"].append({"code": "speed", "message": f"PTP cruise bound TCP {cruise:.0f} mm/s > "
                                                                     f"{p['max_tcp_speed_mm_s']:g} mm/s (model{extra})"})
            exempt = False
        else:
            ori = _options(st["ori_type"])
            if q_prev is None:
                Q = q1[None]
            elif UNKNOWN in ori:
                Q = q1[None]
                nc.append(f"not checked (path): $ORI_TYPE unknown ({_UNK_WHY}); target pose checked")
            elif len(ori) > 1:
                Q = q1[None]
                nc.append("not checked (path): $ORI_TYPE depends on a branch / loop; target pose checked")
            elif (next(iter(ori)) or "#VAR") == "#VAR" and math.degrees(float(rotation_angle_between(
                    self.robot.fk(q_prev)[:3, :3], T1[:3, :3]))) > 180.0 - p.get("lin_ori_ambiguous_deg", 1.0):
                Q = q1[None]
                nc.append("not checked (path): the LIN turns the tool by about 180 deg, so the shortest rotation "
                          "(and the controller's path) is not unique; target pose checked")
            else:
                ori_type = next(iter(ori)) or "#VAR"
                Ts = self._lin_samples(self.robot.fk(q_prev), T1, ori_type)
                if ori_type == "#CONSTANT":
                    rec["notes"].append("$ORI_TYPE #CONSTANT: start orientation held, target ABC ignored")
                Q, prob = self._follow_lin(q_prev, Ts)
                if prob is not None:
                    rec["flags"].append({"code": prob[0], "message": prob[1]})
                if prob is not None and prob[0] == "reach":
                    q1 = None
                else:
                    q1 = Q[-1]
                why = "followed from the start of the LIN"
            exempt = True
        rec["n_samples"] = int(len(Q))
        self._evaluate(rec, m, Q, exempt)
        rec["branch"] = None if q1 is None else {**self._branch(q1), "how": why}
        rec["end_axis_deg"] = None if q1 is None else np.round(np.degrees(q1), 3).tolist()
        return q1


# --------------------------------------------------------------------------- report
ASSUMPTIONS = [
    "Override 100 % everywhere: between RSI_ON and RSI_OFF a PC can set the override to any value in 1..100, "
    "and outside RSI the pendant can; $OV_PRO assignments are listed, not applied.",
    "Frames: only TOOL 3 and BASE 0 of config/cell.yaml are modelled (TOOL 3 copied from the controller, to be "
    "re-confirmed; BASE 0 = robot base). Any other frame number is an error.",
    "Kinematics: the URDF (no meshes); joint limits = URDF ranges minus use_margin_deg; the controller's software "
    "limits are unknown. S / T bits follow Robot.status_bits / turn_bits (unverified on the controller).",
    "PTP TCP speed (model estimate): the M1 flag uses the cruise bound, the TCP speed of the synchronous PTP "
    "with the leading axis at its $VEL_AXIS % of the URDF velocity (URDF axis speeds unverified); it does not "
    "depend on the axis accelerations. The ramp-limited peak of the transit speed model ($ACC_AXIS % of the "
    "ASSUMED axis_acc_deg_s2) is reported for information only. $VEL_AXIS / $ACC_AXIS not set by the program "
    "are assumed at unset_vel_axis_pct / unset_acc_axis_pct. BAS(#PTP_PARAMS, v) sets $VEL_AXIS to the larger "
    "of v and PDAT_ACT.VEL (an upper bound for either use by the controller) and $ACC_AXIS to PDAT_ACT.ACC. "
    "LIN TCP speed = $VEL.CP.",
    "PTP path = straight line in joint space (synchronous PTP); LIN path = straight line with the orientation "
    "interpolated on the shortest rotation (#VAR) or held (#CONSTANT). Blending (C_DIS / C_PTP) is not modelled.",
    "Collision: CollisionModel primitives (capsules sized from the datasheet drawing, tool geometry estimated), "
    "bed box, clamp boxes from photo-estimated positions (+-15-30 mm, inflated), keep-out boxes, floor. The "
    "nozzle tip is exempt from the bed and floor on LIN moves only. Hose, cables and the printed part are not "
    "modelled.",
    "Only an explicit list of statements is accepted as without motion; any other statement, including any "
    "call of a function or subroutine, is not interpreted (exit code 2). After it, every setting it may change "
    "($TOOL, $BASE, $VEL.CP, $VEL_AXIS, $ACC_AXIS; for a call: all of them and the pose) is unknown, and every "
    "later motion that depends on it is not checked until a modelled statement sets it again.",
    "Branches and loops: every IF / SWITCH branch is followed and every loop is run until no new setting state "
    "appears; a motion is checked with the worst case of every value a setting may hold there (highest "
    "$VEL.CP, $VEL_AXIS, $ACC_AXIS; any TOOL / BASE that may be active). A motion whose start pose depends on "
    "a branch is checked from every possible start; one whose start depends on a loop is not checked.",
    "Frame tables: a write to TOOL_DATA[n] / BASE_DATA[n] (a whole frame or one field) is not interpreted "
    "(exit code 2). Frame n is unknown from there on (also the active $TOOL / $BASE if it may be frame n), so "
    "every later Cartesian motion that uses it is not checked, also after BAS(#TOOL, n) / $TOOL = TOOL_DATA[n]; "
    "a call may run any frame write of the file.",
    "DEF sections: each DEF is checked on its own. A DEF after the first one starts with an unknown pose and "
    "unknown settings (they come from the caller), so its first motion is not checked and every setting must "
    "be set inside it before a motion that depends on it is checked.",
    "Values: $VEL.CP, $VEL_AXIS, $ACC_AXIS, BAS(#VEL_PTP / #PTP_PARAMS) and PDAT_ACT.VEL / .ACC must be finite "
    "and > 0; any other value is an error (exit code 2) and leaves the setting unknown. $ORI_TYPE other than "
    "#VAR / #CONSTANT, or unknown, makes the path of a dependent LIN not checked.",
    "Nothing here was run on the robot: 'no flag' means no flag in this model, not a check on the controller.",
]


def build_report(program_path, prog, motions, checker):
    issues = [i.as_dict() for i in prog.issues]
    flags = [{"line": r["line"], **f} for r in motions for f in r["flags"]]
    codes = sorted({f["code"] for f in flags} | {i["code"] for i in issues})
    has_error = any(i["severity"] == "error" for i in issues)
    has_flag = bool(flags) or any(i["severity"] == "flag" for i in issues)
    not_checked = [r for r in motions if not r.get("checked", True)]
    exit_code = 2 if has_error else 1 if has_flag else 3 if not_checked else 0
    stmts = [{"line": s.line, "kind": s.kind, "what": s.what, "text": s.text.strip()} for s in prog.statements]
    sections = []
    for k, sec in enumerate(prog.sections):
        ms = [r for r, m in zip(motions, prog.motions) if m.section == k]
        sections.append({"def": sec["name"], "line": sec["line"], "end_line": sec["end_line"],
                         "start": "program start (--start-axis)" if sec["first"] else
                                  "unknown: pose and settings from the caller",
                         "motions": len(ms), "flagged_motions": sum(bool(r["flags"]) for r in ms),
                         "not_checked_motions": sum(not r.get("checked", True) for r in ms),
                         "issue_lines": [i["line"] for i in issues if sec["line"] <= i["line"] <= sec["end_line"]]})
    ko = {"source": checker.keep_out_source, "boxes_from_file": checker.n_keep_out_extra,
          "boxes_in_cell_yaml": checker.n_keep_out_cell}
    return {
        "program": Path(program_path).name,
        "exit_code": exit_code,
        "result": EXIT_RESULT[exit_code],
        "summary": {
            "motions": len(motions),
            "ptp": sum(r["kind"] == "PTP" for r in motions),
            "lin": sum(r["kind"] == "LIN" for r in motions),
            "flagged_motions": sum(bool(r["flags"]) for r in motions),
            "not_checked_motions": len(not_checked),
            "not_checked_lines": [r["line"] for r in not_checked],
            "flag_counts": {c: sum(f["code"] == c for f in flags) for c in FLAGS if any(f["code"] == c for f in flags)},
            "issue_codes": codes,
            "not_interpreted_lines": [s["line"] for s in stmts if s["kind"] == "not_interpreted"],
            "invalid_setting_lines": [s["line"] for s in stmts if s["kind"] == "invalid_setting"],
            "non_motion_lines": [s["line"] for s in stmts if s["kind"] == "no_motion"],
            "no_effect_setting_lines": [s["line"] for s in stmts if s["kind"] == "setting"
                                        and "no effect on the checks" in s["what"]],
            "max_speed_mm_s": max((r["speed_mm_s"] for r in motions if r.get("speed_mm_s") is not None),
                                  default=None),
            "min_clearance_mm": min((r["min_clearance_mm"] for r in motions if "min_clearance_mm" in r),
                                    default=None),
            "min_tip_height_mm": min((r["min_tip_height_mm"] for r in motions if "min_tip_height_mm" in r),
                                     default=None),
        },
        "sections": sections,
        "keep_out": ko,
        "thresholds": dict(checker.p),
        "assumptions": ASSUMPTIONS,
        "issues": issues,
        "flags": flags,
        "statements": stmts,
        "motions": motions,
    }


def _fmt_target(r):
    t = r["target"]
    return ", ".join(f"{k} {v}" if isinstance(v, int) else f"{k} {f'{v:.4f}'.rstrip('0').rstrip('.')}"
                     for k, v in t.items())


def write_markdown(path, rep):
    s = rep["summary"]
    ko = rep["keep_out"]
    L = [f"# Static check: {rep['program']}", "",
         "Simulation only. Checked against the cell model of this repository (config/cell.yaml, "
         "config/planner.yaml `src_check`). **No flag means no flag in this model** (URDF without meshes, "
         "photo-estimated clamps, unverified axis speeds); it does not replace a check on the controller.", "",
         f"**Result: {rep['result']} (exit code {rep['exit_code']}).**", "",
         "## Summary", "",
         f"- motions: {s['motions']} ({s['ptp']} PTP, {s['lin']} LIN), flagged: {s['flagged_motions']}, "
         f"not checked: {s['not_checked_motions']}",
         "- flags: " + (", ".join(f"{k} {v}" for k, v in s["flag_counts"].items()) or "none"),
         "- errors: " + (", ".join(f"line {i['line']}: {i['message']}" for i in rep["issues"]
                                    if i["severity"] == "error") or "none"),
         f"- max TCP speed: {s['max_speed_mm_s']} mm/s (100 % override); min clearance: {s['min_clearance_mm']} mm; "
         f"min tip height above the highest bed top: {s['min_tip_height_mm']} mm",
         f"- keep-out: {ko['source']}; {ko['boxes_from_file']} box(es) from that file, "
         f"{ko['boxes_in_cell_yaml']} in cell.yaml (coordinates not reported)", "",
         "## DEF sections", "",
         "Each DEF is checked on its own. A DEF after the first one starts with an unknown pose and unknown "
         "settings (they come from the caller), so its first motion is not checked.", ""]
    L += [f"- DEF {d['def']} (lines {d['line']}-{d['end_line']}; start {d['start']}): {d['motions']} motion(s), "
          f"flagged {d['flagged_motions']}, not checked {d['not_checked_motions']}, errors "
          f"{len(d['issue_lines'])}" for d in rep.get("sections", [])] or ["none"]
    L += ["", "## Statements not interpreted or invalid", ""]
    ni = [x for x in rep["statements"] if x["kind"] in ("not_interpreted", "invalid_setting")]
    L += [f"- line {x['line']}: `{x['text']}`" for x in ni] or ["none"]
    L += ["", "## Motions not checked", ""]
    L += [f"- line {r['line']}: " + "; ".join(r["not_checked"]) for r in rep["motions"] if r["not_checked"]] or ["none"]
    L += ["", "## Settings without effect on the checks", ""]
    ne = [x for x in rep["statements"] if x["kind"] == "setting" and "no effect on the checks" in x["what"]]
    L += [f"- line {x['line']}: `{x['text']}` ({x['what']})" for x in ne] or ["none"]
    L += ["", "## Statements without motion", ""]
    nm = [x for x in rep["statements"] if x["kind"] == "no_motion"]
    L += [f"- line {x['line']}: `{x['text']}` ({x['what']})" for x in nm] or ["none"]
    L += ["", "## Flags", ""]
    L += [f"- line {f['line']}: {f['code']}: {f['message']}" for f in rep["flags"]] or ["none"]
    L += ["", "## Motions"]
    defs = rep.get("sections") or [{"def": None}]
    for d in defs:
        L += [""] + ([f"### DEF {d['def']} (lines {d['line']}-{d['end_line']})", ""] if d["def"] is not None else [])
        L += ["| line | kind | target | speed mm/s | min clearance mm (pair) | tip height mm | branch | flags |",
              "|---|---|---|---|---|---|---|---|"]
        L += [_motion_row(r) for r in rep["motions"] if d["def"] is None or d["line"] <= r["line"] <= d["end_line"]]
    _rest(L, rep)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def _motion_row(r):
    br = r.get("branch")
    brs = "-" if not br else f"S {br['S']} T {br['T']}" + (" (assumed)" if br["how"].startswith("ASSUMED") else "")
    cl = "-" if "min_clearance_mm" not in r else f"{r['min_clearance_mm']:.1f} ({' / '.join(r['clearance_pair'])})"
    sp = "-" if r.get("speed_mm_s") is None else f"{r['speed_mm_s']:g}"
    if r.get("speed_mm_s") is None and r.get("speed_source", "").startswith("start unknown"):
        sp = "start unknown, not checked"
    elif r["not_checked"]:
        sp = ("-" if sp == "-" else sp) + " (not checked)"
    th = "-" if "min_tip_height_mm" not in r else f"{r['min_tip_height_mm']:.2f}"
    fl = "; ".join(f"{f['code']}: {f['message']}" for f in r["flags"]) or ""
    return f"| {r['line']} | {r['kind']} | {_fmt_target(r)} | {sp} | {cl} | {th} | {brs} | {fl} |"


def _rest(L, rep):
    notes = [(r["line"], n) for r in rep["motions"] for n in r["notes"]]
    assumed = [r for r in rep["motions"] if r.get("branch") and r["branch"]["how"].startswith("ASSUMED")]
    if assumed or notes:
        L += ["", "## Notes", ""]
        L += [f"- line {r['line']}: branch {r['branch']['how']}" for r in assumed]
        seen = {}
        for ln, n in notes:
            seen.setdefault(n, []).append(ln)
        for n, lns in seen.items():
            L.append(f"- {n} (lines {_ranges(lns)})")
    L += ["", "## Thresholds (config/planner.yaml src_check)", ""]
    L += [f"- {k}: {v}" for k, v in rep["thresholds"].items()]
    L += ["", "## Assumptions", ""] + [f"- {a}" for a in rep["assumptions"]]


def _ranges(lines):
    lines = sorted(set(lines))
    out, a = [], None
    for i, x in enumerate(lines):
        if a is None:
            a = x
        if i + 1 == len(lines) or lines[i + 1] != x + 1:
            out.append(str(a) if a == x else f"{a}-{x}")
            a = None
    return ", ".join(out)


def check_text(text, name="program.src", checker=None, **kw):
    """Parse and check KRL text; returns the report dict (no files written)."""
    checker = checker or SrcChecker(**kw)
    prog = parse_program(text)
    motions = checker.check_program(prog)
    return build_report(name, prog, motions, checker)


_PROTECTED = ("config", "results")


def _protected(path):
    p = Path(path).resolve()
    return any(p == (REPO / d).resolve() or (REPO / d).resolve() in p.parents for d in _PROTECTED)


def output_dir(program, out=None):
    """Where the report goes: ``out`` if given (never inside config/ or results/), else next to the
    program; a program inside config/ or results/ gets out/src_check/<relative path> instead."""
    if out is not None:
        if _protected(out):
            raise ValueError("the checker never writes into config/ or results/; choose another --out")
        return Path(out)
    d = Path(program).resolve().parent
    if _protected(d):
        return REPO / "out" / "src_check" / d.relative_to(REPO.resolve())
    return d


def check_file(program, out=None, keep_out_path=None, use_local_keep_out=True, start_axis_deg=None, cfg=None):
    """Check a .src file and write <stem>_check.md / <stem>_check.json. Returns (report, md, json paths)."""
    extra, src = [], "none (no config/keep_out.local.yaml, no --keep-out)"
    if keep_out_path is not None:
        extra = load_keep_out(keep_out_path)
        src = "file given with --keep-out"
    elif use_local_keep_out and LOCAL_KEEP_OUT.exists():
        extra = load_keep_out(LOCAL_KEEP_OUT)
        src = "local file config/keep_out.local.yaml used"
    checker = SrcChecker(cfg=cfg, keep_out=extra, keep_out_source=src, start_axis_deg=start_axis_deg)
    with open(program, encoding="utf-8") as f:
        text = f.read()
    rep = check_text(text, Path(program).name, checker)
    if start_axis_deg is not None:
        rep["start_axis_deg"] = [float(x) for x in start_axis_deg]
    d = output_dir(program, out)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(program).stem
    md, js = d / f"{stem}_check.md", d / f"{stem}_check.json"
    write_markdown(md, rep)
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    return rep, md, js

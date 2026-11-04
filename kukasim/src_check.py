"""Static check of a KRL program (.src) against the cell model. Simulation only.

The program is read line by line in the dialect that this repository's exporters and
KUKA|prc-style generators write (see ``parse_program``); every other statement is
listed as "not interpreted" with its line number. Every motion is then replayed on
the models that already exist in this repository, nothing is re-modelled here:

* kinematics: ``kinematics.Robot`` (URDF FK, analytic IK, TOOL 3 / BASE 0 of cell.yaml);
* collision: ``collision.CollisionModel`` (capsules, bed / clamp / keep-out boxes, floor);
* PTP sampling and the PTP TCP-speed estimate: ``transit.TransitPlanner``.

Checks per motion (thresholds: ``src_check`` in config/planner.yaml):

1. reach: IK inside the URDF joint limits minus ``use_margin_deg``; a LIN is sampled and
   one IK branch is followed from its start: any configuration (status) change, joint
   step above ``lin_max_joint_step_deg`` or sample without IK is flagged;
2. singularity margins (|A5|, elbow, shoulder) on every sample;
3. speed (M1): LIN at the ``$VEL.CP`` in force, PTP from the transit speed model at the
   ``$VEL_AXIS`` / ``$ACC_AXIS`` in force; always at 100 % override;
4. clearance (M2): minimum signed clearance and its primitive / obstacle pair; the nozzle
   tip is exempt from the bed on LIN moves only;
5. nozzle height (M3): tip Z above the highest bed top, and tip XY of LIN moves inside the
   bed outline minus a margin.

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
ERRORS = ("parse_error", "frame_not_modelled")
FLAGS = ("reach", "lin_branch", "singularity", "speed", "clearance", "nozzle_height", "bed_edge",
         "not_interpreted")

MOTION_WORDS = ("PTP", "LIN", "CIRC", "SPTP", "SLIN", "SCIRC", "PTP_REL", "LIN_REL", "CIRC_REL",
                "SPTP_REL", "SLIN_REL", "SCIRC_REL", "SPL", "SPLINE", "ASYPTP")

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE = {k: re.compile(v, re.IGNORECASE) for k, v in {
    "attribute": r"^&\w+",
    "def": r"^(?:GLOBAL\s+)?DEF\s+\w+\s*\([^)]*\)$",
    "end": r"^END$",
    "halt": r"^HALT$",
    "wait": rf"^WAIT\s+SEC\s+(?:{_NUM}|\w+)$",
    "decl": r"^DECL\s+\w+\s+\w+",
    "if": r"^IF\s+.+\s+THEN$",
    "endif": r"^ENDIF$",
    "bas": rf"^BAS\s*\(\s*#(\w+)\s*,\s*({_NUM})\s*\)$",
    "motion": r"^(PTP|LIN)\s*\{([^}]*)\}\s*(C_DIS|C_PTP)?$",
    "vel_cp": r"^\$VEL\.CP\s*=\s*(.+)$",
    "vel_axis": r"^\$VEL_AXIS\s*\[\s*(\d+)\s*\]\s*=\s*(.+)$",
    "acc_axis": r"^\$ACC_AXIS\s*\[\s*(\d+)\s*\]\s*=\s*(.+)$",
    "apo_cdis": r"^\$APO\.CDIS\s*=\s*(.+)$",
    "ov_pro": r"^\$OV_PRO\s*=\s*(.+)$",
    "base": r"^\$BASE\s*=\s*(?:(\$NULLFRAME)|BASE_DATA\s*\[\s*(\d+)\s*\])$",
    "tool": r"^\$TOOL\s*=\s*TOOL_DATA\s*\[\s*(\d+)\s*\]$",
    "ori_type": r"^\$ORI_TYPE\s*=\s*#(CONSTANT|VAR)$",
    "rsi": r"^(?:\w+\s*=\s*)?RSI_(CREATE|ON|OFF)\s*\(.*\)$",
    "user_assign": r"^([A-Za-z_]\w*)(?:\s*\[[^\]]*\])?\s*=\s*(.+)$",
}.items()}


# --------------------------------------------------------------------------- parsing
@dataclass
class Statement:
    line: int
    text: str
    kind: str          # motion | setting | frame | no_motion | not_interpreted
    what: str = ""


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
    state: dict = field(default_factory=dict)   # settings in force at this motion


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
    issues: list                   # parse / frame errors and not-interpreted flags


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


def parse_program(text):
    """Read a .src text into statements, motions (with the settings in force) and issues."""
    st = {"vel_cp_m_s": None, "vel_axis_pct": [None] * 6, "acc_axis_pct": [None] * 6,
          "apo_cdis": None, "ov_pro": None, "tool": None, "base": None, "ori_type": None,
          "rsi_on": False, "in_if": 0}
    statements, motions, issues = [], [], []

    def add(i, raw, kind, what=""):
        statements.append(Statement(i, raw.rstrip(), kind, what))

    for i, raw in enumerate(text.splitlines(), start=1):
        s = _strip_comment(raw)
        if not s:
            continue
        R = {k: r.match(s) for k, r in _RE.items()}
        if R["attribute"]:
            add(i, raw, "no_motion", "editor attribute")
        elif R["def"]:
            add(i, raw, "no_motion", "DEF")
        elif R["end"]:
            add(i, raw, "no_motion", "END")
        elif R["halt"]:
            add(i, raw, "no_motion", "HALT")
        elif R["wait"]:
            add(i, raw, "no_motion", "WAIT SEC")
        elif R["decl"]:
            add(i, raw, "no_motion", "DECL")
        elif R["if"]:
            st["in_if"] += 1
            add(i, raw, "no_motion", "IF")
        elif R["endif"]:
            st["in_if"] = max(0, st["in_if"] - 1)
            add(i, raw, "no_motion", "ENDIF")
        elif R["rsi"]:
            w = R["rsi"].group(1).upper()
            if w == "ON":
                st["rsi_on"] = True
            elif w == "OFF":
                st["rsi_on"] = False
            add(i, raw, "no_motion", f"RSI_{w}")
        elif R["bas"]:
            what, n = R["bas"].group(1).upper(), float(R["bas"].group(2))
            if what == "TOOL":
                st["tool"] = int(n)
                add(i, raw, "frame", f"TOOL {int(n)}")
                if int(n) != MODELLED_TOOL:
                    issues.append(Issue("frame_not_modelled", i, f"TOOL {int(n)}: frame not modelled "
                                        f"(only TOOL {MODELLED_TOOL} of cell.yaml)"))
            elif what == "BASE":
                st["base"] = int(n)
                add(i, raw, "frame", f"BASE {int(n)}")
                if int(n) != MODELLED_BASE:
                    issues.append(Issue("frame_not_modelled", i, f"BASE {int(n)}: frame not modelled "
                                        f"(only BASE {MODELLED_BASE} of cell.yaml)"))
            elif what == "INITMOV":
                st["vel_cp_m_s"] = None
                st["vel_axis_pct"] = [None] * 6
                st["acc_axis_pct"] = [None] * 6
                add(i, raw, "setting", "BAS(#INITMOV): controller defaults (not modelled: speeds unset)")
            elif what == "VEL_PTP":
                st["vel_axis_pct"] = [n] * 6
                add(i, raw, "setting", f"$VEL_AXIS[1..6] = {n:g} %")
            else:
                add(i, raw, "not_interpreted")
                issues.append(Issue("not_interpreted", i, f"not interpreted: {s}"))
        elif R["motion"]:
            kind, body, approx = R["motion"].group(1).upper(), R["motion"].group(2), R["motion"].group(3)
            try:
                tt, vals, S, T, ext = _parse_aggregate(body)
                if kind == "LIN" and tt == "axis":
                    raise ValueError("LIN to an axis target")
            except ValueError as e:
                add(i, raw, "not_interpreted", "motion")
                issues.append(Issue("parse_error", i, f"motion not interpreted ({e}): {s}"))
                continue
            add(i, raw, "motion", kind)
            motions.append(Motion(i, s, kind, tt, vals, S, T, ext, approx.upper() if approx else None,
                                  {k: (list(v) if isinstance(v, list) else v) for k, v in st.items()}))
        elif R["vel_cp"]:
            v = _num(R["vel_cp"].group(1))
            st["vel_cp_m_s"] = v
            if v is None:
                add(i, raw, "not_interpreted")
                issues.append(Issue("not_interpreted", i, f"$VEL.CP is not a number (speed unknown): {s}"))
            else:
                add(i, raw, "setting", f"$VEL.CP = {v:g} m/s")
        elif R["vel_axis"] or R["acc_axis"]:
            key = "vel_axis_pct" if R["vel_axis"] else "acc_axis_pct"
            m = R["vel_axis"] or R["acc_axis"]
            k, v = int(m.group(1)), _num(m.group(2))
            if not 1 <= k <= 6 or v is None:
                add(i, raw, "not_interpreted")
                issues.append(Issue("not_interpreted", i, f"not interpreted: {s}"))
                if 1 <= k <= 6:
                    st[key][k - 1] = None
            else:
                st[key][k - 1] = v
                add(i, raw, "setting", f"{'$VEL_AXIS' if key == 'vel_axis_pct' else '$ACC_AXIS'}[{k}] = {v:g} %")
        elif R["apo_cdis"]:
            st["apo_cdis"] = _num(R["apo_cdis"].group(1))
            add(i, raw, "setting", "$APO.CDIS (blending not modelled)")
        elif R["ov_pro"]:
            st["ov_pro"] = R["ov_pro"].group(1).strip()
            add(i, raw, "no_motion", "$OV_PRO (speeds are checked at 100 % override)")
        elif R["base"]:
            n = MODELLED_BASE if R["base"].group(1) else int(R["base"].group(2))
            st["base"] = n
            add(i, raw, "frame", f"BASE {n}")
            if n != MODELLED_BASE:
                issues.append(Issue("frame_not_modelled", i, f"BASE {n}: frame not modelled "
                                    f"(only BASE {MODELLED_BASE} of cell.yaml)"))
        elif R["tool"]:
            n = int(R["tool"].group(1))
            st["tool"] = n
            add(i, raw, "frame", f"TOOL {n}")
            if n != MODELLED_TOOL:
                issues.append(Issue("frame_not_modelled", i, f"TOOL {n}: frame not modelled "
                                    f"(only TOOL {MODELLED_TOOL} of cell.yaml)"))
        elif R["ori_type"]:
            st["ori_type"] = "#" + R["ori_type"].group(1).upper()
            add(i, raw, "setting", f"$ORI_TYPE = {st['ori_type']}")
        elif R["user_assign"] and not s.startswith("$"):
            add(i, raw, "no_motion", f"assignment to user variable {R['user_assign'].group(1)}")
        else:
            first = re.match(r"[A-Za-z_$]\w*", s)
            if first and first.group(0).upper() in MOTION_WORDS:
                add(i, raw, "not_interpreted", "motion")
                issues.append(Issue("parse_error", i, f"motion not interpreted: {s}"))
            else:
                add(i, raw, "not_interpreted")
                issues.append(Issue("not_interpreted", i, f"not interpreted: {s}"))
    return Program(statements, motions, issues)


# --------------------------------------------------------------------------- keep-out
def load_keep_out(path):
    """Boxes from a keep-out file: a list, or a mapping with key ``keep_out`` (same entry
    format as ``keep_out`` in cell.yaml)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    if d is None:
        return []
    if isinstance(d, dict):
        d = d.get("keep_out") or []
    if not isinstance(d, list):
        raise ValueError(f"{Path(path).name}: expected a list of keep-out boxes")
    return d


# --------------------------------------------------------------------------- checker
@dataclass
class _Track:
    q: np.ndarray | None = None      # current joints (rad); None = unknown


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

    def _axis_pct(self, vals, unset):
        return np.array([unset if v is None else float(v) for v in vals])

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

    def check_program(self, prog: Program):
        p = self.p
        tr = _Track(None if self.start_q is None else self.start_q.copy())
        out = []
        for m in prog.motions:
            st = m.state
            rec = {"line": m.line, "kind": m.kind, "target_type": m.target_type,
                   "target": {k: v for k, v in m.values.items()}, "approx": m.approx,
                   "flags": [], "notes": []}
            if m.s is not None:
                rec["target"]["S"] = m.s
            if m.t is not None:
                rec["target"]["T"] = m.t
            if m.external:
                rec["notes"].append("external axes " + ", ".join(sorted(m.external)) + " not modelled")
            if m.approx:
                rec["notes"].append(f"{m.approx} blending not modelled (checked through the programmed point)")
            if st["in_if"]:
                rec["notes"].append("inside IF ... ENDIF: checked as if executed")
            if st["rsi_on"]:
                rec["notes"].append("RSI on: a PC can set the override (1..100 %) and RSI corrections are "
                                    "not modelled")
            out.append(rec)
            q_prev = tr.q
            rec["start_known"] = q_prev is not None
            # frames
            if m.target_type == "cartesian" and (st["tool"] != MODELLED_TOOL or st["base"] != MODELLED_BASE):
                if st["tool"] is None or st["base"] is None:
                    missing = " and ".join(n for n, v in (("TOOL", st["tool"]), ("BASE", st["base"])) if v is None)
                    prog.issues.append(Issue("frame_not_modelled", m.line,
                                             f"{missing} not set before this Cartesian motion"))
                rec["notes"].append("not checked: frame not modelled")
                rec["speed_mm_s"] = None
                tr.q = None
                continue
            # speed settings in force
            if m.kind == "LIN":
                v = st["vel_cp_m_s"]
                rec["speed_mm_s"] = None if v is None else round(v * 1000.0, 3)
                rec["speed_source"] = "$VEL.CP at 100 % override"
                if v is None:
                    rec["flags"].append({"code": "speed", "message": "$VEL.CP not set by the program "
                                                                     "(controller default unknown)"})
                elif v * 1000.0 > p["max_tcp_speed_mm_s"]:
                    rec["flags"].append({"code": "speed", "message": f"$VEL.CP {v * 1000:.0f} mm/s > "
                                                                     f"{p['max_tcp_speed_mm_s']:g} mm/s"})
            vel = self._axis_pct(st["vel_axis_pct"], p["unset_vel_axis_pct"])
            acc = self._axis_pct(st["acc_axis_pct"], p["unset_acc_axis_pct"])
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
                    tr.q = None
                    if m.kind == "PTP":
                        rec["speed_mm_s"] = None
                        rec["speed_source"] = "not checked (target not reachable)"
                    continue
            # path
            if m.kind == "PTP":
                if q_prev is None:
                    Q = q1[None]
                    rec["speed_mm_s"] = None
                    rec["speed_source"] = "start unknown, not checked (path and speed); target pose checked"
                else:
                    Q = self.tpl.ptp_samples(q_prev, q1, p["ptp_sample_deg"])
                    peak, cruise, _ = self.tpl.ptp_speed(Q, vel, acc)
                    rec["speed_mm_s"] = round(peak, 1)
                    rec["speed_source"] = ("PTP peak TCP speed, transit speed model (model estimate) at $VEL_AXIS "
                                           + "/".join(f"{x:g}" for x in vel) + " %, $ACC_AXIS "
                                           + "/".join(f"{x:g}" for x in acc) + " %")
                    if peak > p["max_tcp_speed_mm_s"]:
                        rec["flags"].append({"code": "speed", "message": f"PTP peak TCP {peak:.0f} mm/s > "
                                                                         f"{p['max_tcp_speed_mm_s']:g} mm/s (model)"})
                exempt = False
            else:
                if q_prev is None:
                    Q = q1[None]
                    rec["notes"].append("start unknown: LIN path not checked, target pose only")
                else:
                    Ts = self._lin_samples(self.robot.fk(q_prev), T1, st["ori_type"] or "#VAR")
                    if st["ori_type"] == "#CONSTANT":
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
            tr.q = q1
        return out


# --------------------------------------------------------------------------- report
ASSUMPTIONS = [
    "Override 100 % everywhere: between RSI_ON and RSI_OFF a PC can set the override to any value in 1..100, "
    "and outside RSI the pendant can; $OV_PRO assignments are listed, not applied.",
    "Frames: only TOOL 3 and BASE 0 of config/cell.yaml are modelled (TOOL 3 copied from the controller, to be "
    "re-confirmed; BASE 0 = robot base). Any other frame number is an error.",
    "Kinematics: the URDF (no meshes); joint limits = URDF ranges minus use_margin_deg; the controller's software "
    "limits are unknown. S / T bits follow Robot.status_bits / turn_bits (unverified on the controller).",
    "PTP TCP speed: transit speed model (model estimate): each axis at most $VEL_AXIS % of its URDF velocity "
    "(URDF axis speeds unverified) and $ACC_AXIS % of the assumed axis_acc_deg_s2; $VEL_AXIS / $ACC_AXIS not set "
    "by the program are assumed at unset_vel_axis_pct / unset_acc_axis_pct. LIN TCP speed = $VEL.CP.",
    "PTP path = straight line in joint space (synchronous PTP); LIN path = straight line with the orientation "
    "interpolated on the shortest rotation (#VAR) or held (#CONSTANT). Blending (C_DIS / C_PTP) is not modelled.",
    "Collision: CollisionModel primitives (capsules sized from the datasheet drawing, tool geometry estimated), "
    "bed box, clamp boxes from photo-estimated positions (+-15-30 mm, inflated), keep-out boxes, floor. The "
    "nozzle tip is exempt from the bed and floor on LIN moves only. Hose, cables and the printed part are not "
    "modelled.",
    "Statements listed as not interpreted are assumed not to move the robot or change the speed settings.",
    "Nothing here was run on the robot: 'no flag' means no flag in this model, not a check on the controller.",
]


def build_report(program_path, prog, motions, checker):
    issues = [i.as_dict() for i in prog.issues]
    flags = [{"line": r["line"], **f} for r in motions for f in r["flags"]]
    codes = sorted({f["code"] for f in flags} | {i["code"] for i in issues})
    has_error = any(i["severity"] == "error" for i in issues)
    has_flag = bool(flags) or any(i["severity"] == "flag" for i in issues)
    exit_code = 2 if has_error else (1 if has_flag else 0)
    stmts = [{"line": s.line, "kind": s.kind, "what": s.what, "text": s.text.strip()} for s in prog.statements]
    ko = {"source": checker.keep_out_source, "boxes_from_file": checker.n_keep_out_extra,
          "boxes_in_cell_yaml": checker.n_keep_out_cell}
    return {
        "program": Path(program_path).name,
        "exit_code": exit_code,
        "result": {0: "no flag (in this model)", 1: "flagged", 2: "parse or frame error"}[exit_code],
        "summary": {
            "motions": len(motions),
            "ptp": sum(r["kind"] == "PTP" for r in motions),
            "lin": sum(r["kind"] == "LIN" for r in motions),
            "flagged_motions": sum(bool(r["flags"]) for r in motions),
            "flag_counts": {c: sum(f["code"] == c for f in flags) for c in FLAGS if any(f["code"] == c for f in flags)},
            "issue_codes": codes,
            "not_interpreted_lines": [s["line"] for s in stmts if s["kind"] == "not_interpreted"],
            "non_motion_lines": [s["line"] for s in stmts if s["kind"] == "no_motion"],
            "max_speed_mm_s": max((r["speed_mm_s"] for r in motions if r.get("speed_mm_s") is not None),
                                  default=None),
            "min_clearance_mm": min((r["min_clearance_mm"] for r in motions if "min_clearance_mm" in r),
                                    default=None),
            "min_tip_height_mm": min((r["min_tip_height_mm"] for r in motions if "min_tip_height_mm" in r),
                                     default=None),
        },
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
         f"- motions: {s['motions']} ({s['ptp']} PTP, {s['lin']} LIN), flagged: {s['flagged_motions']}",
         "- flags: " + (", ".join(f"{k} {v}" for k, v in s["flag_counts"].items()) or "none"),
         "- errors: " + (", ".join(f"line {i['line']}: {i['message']}" for i in rep["issues"]
                                    if i["severity"] == "error") or "none"),
         f"- max TCP speed: {s['max_speed_mm_s']} mm/s (100 % override); min clearance: {s['min_clearance_mm']} mm; "
         f"min tip height above the highest bed top: {s['min_tip_height_mm']} mm",
         f"- keep-out: {ko['source']}; {ko['boxes_from_file']} box(es) from that file, "
         f"{ko['boxes_in_cell_yaml']} in cell.yaml (coordinates not reported)", "",
         "## Statements not interpreted", ""]
    ni = [x for x in rep["statements"] if x["kind"] == "not_interpreted"]
    L += [f"- line {x['line']}: `{x['text']}`" for x in ni] or ["none"]
    L += ["", "## Statements without motion", ""]
    nm = [x for x in rep["statements"] if x["kind"] == "no_motion"]
    L += [f"- line {x['line']}: `{x['text']}` ({x['what']})" for x in nm] or ["none"]
    L += ["", "## Flags", ""]
    L += [f"- line {f['line']}: {f['code']}: {f['message']}" for f in rep["flags"]] or ["none"]
    L += ["", "## Motions", "",
          "| line | kind | target | speed mm/s | min clearance mm (pair) | tip height mm | branch | flags |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep["motions"]:
        br = r.get("branch")
        brs = "-" if not br else f"S {br['S']} T {br['T']}" + (" (assumed)" if br["how"].startswith("ASSUMED") else "")
        cl = "-" if "min_clearance_mm" not in r else f"{r['min_clearance_mm']:.1f} ({' / '.join(r['clearance_pair'])})"
        sp = "-" if r.get("speed_mm_s") is None else f"{r['speed_mm_s']:g}"
        if r.get("speed_mm_s") is None and r.get("speed_source", "").startswith("start unknown"):
            sp = "start unknown, not checked"
        th = "-" if "min_tip_height_mm" not in r else f"{r['min_tip_height_mm']:.2f}"
        fl = "; ".join(f"{f['code']}: {f['message']}" for f in r["flags"]) or ""
        L.append(f"| {r['line']} | {r['kind']} | {_fmt_target(r)} | {sp} | {cl} | {th} | {brs} | {fl} |")
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


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
    with open(program) as f:
        text = f.read()
    rep = check_text(text, Path(program).name, checker)
    if start_axis_deg is not None:
        rep["start_axis_deg"] = [float(x) for x in start_axis_deg]
    d = output_dir(program, out)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(program).stem
    md, js = d / f"{stem}_check.md", d / f"{stem}_check.json"
    write_markdown(md, rep)
    with open(js, "w", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    return rep, md, js

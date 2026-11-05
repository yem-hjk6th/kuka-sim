"""Moves between printed segments of a KRL export: retract, transit, approach (issue #2).

Simulation only. Every transit is planned and checked here before it is written:

* **Retract / approach.** LIN along the nozzle axis, away from the print (tool -X:
  in this cell's TOOL 3 the nozzle points along tool +X, KUKA's tool direction),
  up to the transit height ``max(bed top + safe_height, higher segment end + min_retract,
  highest Z printed so far + printed_clearance)``: the PTP must clear material already
  printed anywhere on the bed, not only the two segment ends. The joint-space PTP
  sags between its ends, so every PTP sample is checked against that clearance and
  the transit is lifted by the deficit if it dips below.
  The orientation is held, so the IK configuration does not change on these LINs;
  they are sampled in Cartesian space with IK continuity and checked with the
  printing collision rule (tip exempt from the bed, which it starts on).
* **Transit.** One PTP (joint-interpolated) move between the retract pose and the
  pose above the next segment start. A KUKA synchronous PTP moves every axis on
  the same normalised profile, so its geometric path is the straight line in
  joint space; it is sampled there and every sample is checked against the bed,
  the clamps and the floor *with* the tip (nothing may be near the bed in transit).
  If it collides, the PTP is lifted to the fallback ``via_heights_mm`` with
  vertical LINs up and down. If nothing is clear, ``TransitError`` is raised:
  an unchecked transit is never returned.
* **Speed (model estimate, enforced).** Each axis is assumed never to exceed
  ``$VEL_AXIS`` % of its URDF velocity (URDF velocities are unverified, see
  cell.yaml). The leading axis sets the cruise path rate ds/dt <= min_j(v_j / |dq_j|);
  the rate ramps up and down with ds2/dt2 = min_j(a_j / |dq_j|), a_j = ``$ACC_AXIS`` %
  of the assumed ``axis_acc_deg_s2`` (a trapezoid, or a triangle on a move too short
  to reach cruise). The peak TCP speed is max over the move of |dx/ds| * ds/dt.
  Before every transit PTP the export sets
  ``$VEL_AXIS[i] = floor(vel_axis_pct * min(1, k * max_tcp / peak))`` (at least 1 %),
  k = ``speed_safety_factor`` (0.8: the URDF axis maxima are unverified), lowered
  further while the model peak is still above k * max_tcp. If it is still above at
  1 %, ``TransitError`` is raised unless ``allow_fast_transit`` is set (then it is
  flagged). The controller's real profile (jerk limits, dynamic model, override) is
  not modelled. LINs run at the programmed $VEL.CP with constant orientation.
* **Hose / cable wrap.** A transit that turns A4 or A6 by more than
  ``max_wrist_turn_deg`` (180) is flagged: the hose and cables on the wrist are not modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rotations import make_transform


class TransitError(RuntimeError):
    """No collision-free transit was found; the export must not continue."""


@dataclass
class Move:
    kind: str                # "LIN" or "PTP"
    label: str               # retract / lift / transit / lower / approach
    q: np.ndarray            # (n, 6) rad, sampled along the move (first = start, last = target)
    tcp: np.ndarray          # (n, 3) mm, TCP positions of the samples
    checked_tip: bool        # True: the tip was checked against the bed too
    clearance_mm: float = np.inf      # min signed clearance (>= 0 collision-free)
    blocking: str | None = None       # obstacle with the smallest clearance, if < 0
    hits: list = field(default_factory=list)   # every obstacle with clearance < 0
    bed_distance_mm: float = np.inf   # min distance of any primitive (tip included) to the bed
    peak_tcp_mm_s: float = 0.0        # model estimate (trapezoid / triangle profile)
    cruise_tcp_mm_s: float = 0.0      # model estimate at cruise rate (reached only on long enough moves)
    duration_s: float = 0.0           # cruise-only lower bound (model estimate)
    vel_axis_pct: float | None = None  # PTP only: $VEL_AXIS[1..6] emitted before this move
    continuous: bool = True           # LIN only: IK continuous along the line


@dataclass
class Transit:
    seg_from: int
    seg_to: int
    z_mm: float                       # TCP height of the PTP
    z_printed_mm: float = -np.inf     # highest Z printed before this transit
    moves: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    tried: list = field(default_factory=list)   # [(z, reason)] candidates that were rejected

    @property
    def ptp(self):
        return next(m for m in self.moves if m.kind == "PTP")

    @property
    def min_clearance_mm(self):
        return min(m.clearance_mm for m in self.moves)

    @property
    def peak_tcp_mm_s(self):
        return max(m.peak_tcp_mm_s for m in self.moves)

    @property
    def joint_change_deg(self):
        """Joint change over the whole transit (retract start -> approach end)."""
        return np.degrees(self.moves[-1].q[-1] - self.moves[0].q[0])

    @property
    def move_clearance_mm(self):
        """Min clearance per move, keyed by label (retract / lift / transit / lower / approach)."""
        return {m.label: float(m.clearance_mm) for m in self.moves}

    @property
    def min_bed_distance_mm(self):
        """Over the moves at transit height (retract / approach start on the print)."""
        return min((m.bed_distance_mm for m in self.moves if m.checked_tip), default=np.inf)

    def summary(self):
        p = self.ptp
        return {"from_segment": self.seg_from, "to_segment": self.seg_to, "z_mm": round(self.z_mm, 2),
                "moves": [f"{m.kind} {m.label}" for m in self.moves],
                "ptp_joint_change_deg": np.round(np.degrees(p.q[-1] - p.q[0]), 2).tolist(),
                "transit_joint_change_deg": np.round(self.joint_change_deg, 2).tolist(),
                "printed_max_z_mm": None if not np.isfinite(self.z_printed_mm) else round(float(self.z_printed_mm), 2),
                "min_clearance_mm": round(float(self.min_clearance_mm), 2),
                "move_min_clearance_mm": {k: round(v, 2) for k, v in self.move_clearance_mm.items()},
                "min_bed_distance_mm": round(float(self.min_bed_distance_mm), 2),
                "ptp_vel_axis_pct": p.vel_axis_pct,
                "peak_tcp_mm_s_model": round(float(self.peak_tcp_mm_s), 1),
                "ptp_cruise_tcp_mm_s_model": round(float(p.cruise_tcp_mm_s), 1),
                "ptp_duration_s_model": round(float(p.duration_s), 1),
                "flags": list(self.flags),
                "rejected": [{"z_mm": round(z, 2), "reason": r} for z, r in self.tried]}


class TransitPlanner:
    @classmethod
    def from_config(cls, robot, cfg, collision=None):
        from .collision import CollisionModel
        m = np.radians(cfg.limit_margin_deg)
        return cls(robot, cfg, collision or CollisionModel(robot, cfg), robot.limits + np.array([m, -m]))

    def __init__(self, robot, cfg, collision, hard_limits):
        self.robot = robot
        self.cfg = cfg
        self.coll = collision
        self.lim = hard_limits
        self.p = cfg.planner["transit"]
        self.bed_top = cfg.bed()["top_z"]
        self.bed_gap = float(cfg.planner["collision"]["bed_gap_mm"])
        self.print_speed = cfg.print_speed

    # ------------------------------------------------------------ speed
    def axis_speed(self, pct):
        return self.robot.vmax * pct / 100.0

    def axis_acc(self, pct=None):
        """Assumed axis accelerations (rad/s^2) at $ACC_AXIS = pct % (default acc_axis_pct)."""
        a = np.radians(np.broadcast_to(np.asarray(self.p["axis_acc_deg_s2"], float), (6,)))
        return a * (self.p["acc_axis_pct"] if pct is None else pct) / 100.0

    def ptp_speed(self, q, pct, acc_pct=None):
        """(peak, cruise) TCP speed (mm/s) and cruise duration (s) of a PTP sampled as q (n,6); model estimate.

        peak: the path rate follows a trapezoid in s (cruise rate from the leading axis' speed,
        ramps from the leading axis' acceleration), so a move too short to reach cruise is
        capped by its distance; cruise: the rate if cruise were reached everywhere."""
        dq = q[-1] - q[0]
        moving = np.abs(dq) > 1e-12
        if not moving.any():
            return 0.0, 0.0, 0.0
        sdot = np.min(self.axis_speed(pct)[moving] / np.abs(dq[moving]))   # 1/s, path parameter rate
        sacc = np.min(self.axis_acc(acc_pct)[moving] / np.abs(dq[moving]))        # 1/s^2
        n = len(q) - 1
        tcp = self.robot.fk(q)[:, :3, 3]
        dxds = np.linalg.norm(np.diff(tcp, axis=0), axis=1) * n
        s_mid = (np.arange(n) + 0.5) / n
        rate = np.minimum(sdot, np.sqrt(2.0 * sacc * np.minimum(s_mid, 1.0 - s_mid)))
        return float((dxds * rate).max()), float(dxds.max() * sdot), float(1.0 / sdot)

    @property
    def speed_limit(self):
        """Enforced model peak for transit PTPs: k * max_tcp_speed_mm_s."""
        return float(self.p["speed_safety_factor"]) * float(self.p["max_tcp_speed_mm_s"])

    def _set_speed(self, m, pct, acc_pct=None):
        m.vel_axis_pct = pct
        m.peak_tcp_mm_s, m.cruise_tcp_mm_s, m.duration_s = self.ptp_speed(m.q, pct, acc_pct)

    def limit_speed(self, m):
        """Lower the PTP's $VEL_AXIS until its model peak is <= speed_limit (or 1 %)."""
        lim = self.speed_limit
        pct = m.vel_axis_pct
        if m.peak_tcp_mm_s > lim:
            new = max(1, math.floor(pct * min(1.0, lim / m.peak_tcp_mm_s)))
            self._set_speed(m, new)
            # a ramp-limited peak falls slower than linearly with $VEL_AXIS: step down if needed
            while m.peak_tcp_mm_s > lim and m.vel_axis_pct > 1:
                self._set_speed(m, m.vel_axis_pct - 1)
        return m

    # ------------------------------------------------------------ checks
    def _check(self, m: Move):
        c, det = self.coll.clearance(m.q, return_detail=True, exempt_tip=not m.checked_tip)
        m.clearance_mm = float(c.min())
        if m.clearance_mm < 0:
            worst = {k: float(v.min()) for k, v in det.items()}
            m.blocking = min(worst, key=worst.get)
            m.hits = sorted((k for k, v in worst.items() if v < 0), key=worst.get)
        if m.checked_tip:
            # bed term includes the bed gap; add it back for the plain distance
            m.bed_distance_mm = float(det["bed"].min() + self.bed_gap)
        return m

    def ptp_samples(self, q0, q1, step_deg=None):
        """Joint-space samples (n, 6) of a synchronous PTP from q0 to q1 (rad), at most
        ``step_deg`` (default ``ptp_sample_deg``) per joint between samples."""
        step = self.p["ptp_sample_deg"] if step_deg is None else step_deg
        n = int(np.ceil(np.degrees(np.abs(q1 - q0)).max() / step)) + 1
        s = np.linspace(0.0, 1.0, max(n, 2))[:, None]
        return q0 + s * (q1 - q0)

    def ptp(self, q0, q1, label="transit", pct=None, checked_tip=True, enforce=False, acc_pct=None):
        """PTP sampled in joint space. enforce: lower $VEL_AXIS to meet speed_limit (transits)."""
        q = self.ptp_samples(q0, q1)
        m = Move("PTP", label, q, self.robot.fk(q)[:, :3, 3], checked_tip)
        self._set_speed(m, self.p["vel_axis_pct"] if pct is None else pct, acc_pct)
        if enforce:
            self.limit_speed(m)
        return self._check(m)

    def lin(self, q0, p1, label, checked_tip):
        """LIN from joint state q0 to TCP position p1, orientation held."""
        T0 = self.robot.fk(q0)
        p0 = T0[:3, 3]
        n = int(np.ceil(np.linalg.norm(p1 - p0) / self.p["lin_sample_mm"])) + 1
        s = np.linspace(0.0, 1.0, max(n, 2))
        P = p0 + s[:, None] * (p1 - p0)
        T = make_transform(np.broadcast_to(T0[:3, :3], (len(s), 3, 3)), P)
        Q, ok = self.robot.ik(T, limits=self.lim)
        q = np.empty((len(s), 6))
        q[0] = q0
        thr = np.radians(self.p["lin_max_joint_step_deg"])
        cont = True
        for i in range(1, len(s)):
            if not ok[i].any():
                cont = False
                q = q[:i]
                break
            d = np.abs(Q[i] - q[i - 1]).max(-1)
            d[~ok[i]] = np.inf
            k = int(np.argmin(d))
            q[i] = Q[i, k]
            if d[k] > thr:
                cont = False
        m = Move("LIN", label, q, P[:len(q)], checked_tip)
        m.continuous = cont and len(q) == len(s)
        m.peak_tcp_mm_s = self.print_speed
        m.duration_s = float(np.linalg.norm(p1 - p0) / self.print_speed)
        return self._check(m)

    # ------------------------------------------------------------ plan
    def retract_point(self, p, d, z):
        """Point on the line p - t d (t >= 0, back along the nozzle) at height z."""
        if d[2] > -0.2:
            raise TransitError(f"nozzle direction {np.round(d, 3)} too flat for a retract along the nozzle axis")
        t = (z - p[2]) / (-d[2])
        return p - max(t, 0.0) * d

    def base_height(self, p_end, p_start, z_printed=-np.inf):
        """Transit height: safe height above the bed, min_retract above both segment ends,
        and printed_clearance above the highest material printed so far (z_printed)."""
        return max(self.bed_top + self.p["safe_height_mm"],
                   max(p_end[2], p_start[2]) + self.p["min_retract_mm"],
                   z_printed + self.p["printed_clearance_mm"])

    def lift(self, q, p, d, z):
        """retract (or, reversed, approach) moves from the print pose q down at p to height z."""
        return self.lin(q, self.retract_point(p, d, z), "retract", checked_tip=False)

    def plan(self, q_end, p_end, d_end, q_start, p_start, d_start, seg_from=0, seg_to=1, z_printed=None):
        """z_printed: highest Z printed before this transit (default: the end of the last segment)."""
        zp = float(p_end[2] if z_printed is None else max(z_printed, p_end[2]))
        z0 = self.base_height(p_end, p_start, zp)
        tr = Transit(seg_from, seg_to, z0, zp)
        r = self.lift(q_end, p_end, d_end, z0)
        a = self.lift(q_start, p_start, d_start, z0)      # computed upwards, emitted reversed
        a.label = "approach"
        for m, what in ((r, "retract"), (a, "approach")):
            if not m.continuous:
                raise TransitError(f"transit {seg_from}->{seg_to}: {what} LIN is not continuous in IK "
                                   "(configuration change or limit on the way up)")
            if m.clearance_mm < 0:
                raise TransitError(f"transit {seg_from}->{seg_to}: {what} LIN collides with {m.blocking} "
                                   f"(clearance {m.clearance_mm:.1f} mm)")
        a = _reverse(a)
        heights = [z0] + [self.bed_top + h for h in self.p["via_heights_mm"] if self.bed_top + h > z0]
        z_clear = zp + self.p["printed_clearance_mm"]       # every PTP sample must stay above this
        k = 0
        retried = set()
        while k < len(heights):
            z = heights[k]
            k += 1
            moves = [r]
            qa, qb = r.q[-1], a.q[0]
            if z > z0 + 1e-9:
                up = self.lin(qa, np.r_[r.tcp[-1][:2], z], "lift", checked_tip=True)
                dn = self.lin(qb, np.r_[a.tcp[0][:2], z], "lower", checked_tip=True)
                if not (up.continuous and dn.continuous):
                    tr.tried.append((z, "vertical LIN not continuous in IK"))
                    continue
                if min(up.clearance_mm, dn.clearance_mm) < 0:
                    tr.tried.append((z, f"vertical LIN collides with {up.blocking or dn.blocking}"))
                    continue
                dn = _reverse(dn)
                dn.label = "lower"
                moves.append(up)
                qa, qb = up.q[-1], dn.q[0]
            ptp = self.ptp(qa, qb, enforce=True)
            sag = z_clear - float(ptp.tcp[:, 2].min())
            if sag > 1e-6:
                # the joint-space PTP dips between its ends: retry once, higher by the deficit
                tr.tried.append((z, f"PTP sags {sag:.1f} mm below the printed-material clearance"))
                if z not in retried:
                    retried.add(z + sag + 1.0)
                    heights.insert(k, z + sag + 1.0)
                continue
            if ptp.clearance_mm < 0:
                i = int(np.argmin(self.coll.clearance(ptp.q, exempt_tip=False)))
                tr.tried.append((z, f"PTP collides with {ptp.blocking} "
                                    f"(clearance {ptp.clearance_mm:.1f} mm at {100 * i / (len(ptp.q) - 1):.0f} % of the move)"))
                continue
            if ptp.peak_tcp_mm_s > self.speed_limit and not self.p["allow_fast_transit"]:
                raise TransitError(f"transit {seg_from}->{seg_to}: model peak TCP speed {ptp.peak_tcp_mm_s:.0f} mm/s "
                                   f"at $VEL_AXIS = {ptp.vel_axis_pct:g} % is above {self.speed_limit:g} mm/s "
                                   f"(= {self.p['speed_safety_factor']:g} x {self.p['max_tcp_speed_mm_s']:g}); "
                                   "set transit.allow_fast_transit to export it flagged")
            moves.append(ptp)
            if z > z0 + 1e-9:
                moves.append(dn)
            moves.append(a)
            tr.z_mm = z
            tr.moves = moves
            tr.flags = self.flags(tr)
            return tr
        raise TransitError(f"transit {seg_from}->{seg_to}: no collision-free transit at heights "
                           f"{', '.join(f'{z:.0f}' for z in heights)} mm: "
                           + "; ".join(f"z {z:.0f}: {why}" for z, why in tr.tried))

    def flags(self, tr):
        f = []
        lim = self.speed_limit
        if tr.peak_tcp_mm_s > lim:
            f.append(f"TCP speed {tr.peak_tcp_mm_s:.0f} mm/s > {lim:g} mm/s (model estimate, allowed by "
                     "allow_fast_transit)")
        turn = self.p["max_wrist_turn_deg"]
        dq = tr.joint_change_deg
        for j in (3, 5):
            if abs(dq[j]) > turn:
                f.append(f"A{j + 1} turns {dq[j]:+.0f} deg (> {turn:g} deg; hose / cable wrap not modelled)")
        if tr.min_clearance_mm < 0:
            f.append(f"enters a collision zone (clearance {tr.min_clearance_mm:.1f} mm)")
        if tr.min_bed_distance_mm < self.p["bed_clearance_mm"]:
            f.append(f"{tr.min_bed_distance_mm:.1f} mm from the bed at transit height "
                     f"(< {self.p['bed_clearance_mm']:g} mm)")
        return f

    # --------------------------------------------- the old move (issue #2)
    def direct_ptp(self, q_end, q_start):
        """The pre-fix move: PTP straight from the last printed point to the next start,
        at the assumed controller default axis speed."""
        # the old export set no $ACC_AXIS either: full acceleration assumed for the before numbers
        return self.ptp(q_end, q_start, "direct", pct=self.p["before_vel_axis_pct"], acc_pct=100,
                        checked_tip=True)


def _reverse(m: Move) -> Move:
    m.q = m.q[::-1].copy()
    m.tcp = m.tcp[::-1].copy()
    return m


def plan_transits(tplanner, res, rows):
    """Transits between consecutive segments of a plan (one per segment break)."""
    tp = res.toolpath
    out = []
    for k in range(1, len(res.segments)):
        e = res.segments[k - 1][1]
        s = res.segments[k][0]
        out.append(tplanner.plan(res.q[e], rows[e][0], tp.dirs[e], res.q[s], rows[s][0], tp.dirs[s], k - 1, k,
                                 z_printed=printed_max_z(rows, res.segments[:k])))
    return out


def printed_max_z(rows, segments):
    """Highest TCP Z over the printed points of the given segments."""
    return max(rows[i][0][2] for a, b in segments for i in range(a, b + 1))


def end_moves(tplanner, res, rows):
    """Approach above the first segment start and retract after the last segment end."""
    tp = res.toolpath
    if not res.segments:
        return None, None
    s, e = res.segments[0][0], res.segments[-1][1]
    zs = tplanner.base_height(rows[s][0], rows[s][0])
    ze = tplanner.base_height(rows[e][0], rows[e][0], printed_max_z(rows, res.segments))
    a = tplanner.lift(res.q[s], rows[s][0], tp.dirs[s], zs)
    r = tplanner.lift(res.q[e], rows[e][0], tp.dirs[e], ze)
    for m, what in ((a, "first approach"), (r, "final retract")):
        if not m.continuous or m.clearance_mm < 0:
            raise TransitError(f"{what} LIN is not clear (continuous {m.continuous}, "
                               f"clearance {m.clearance_mm:.1f} mm, {m.blocking})")
    a = _reverse(a)
    a.label = "approach"
    return a, r

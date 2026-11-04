"""Speed of the first PTP of a generated program, which starts from an unknown pose.

Simulation only. Every generated KRL program begins with a PTP from wherever the
robot happens to be. Its TCP speed is bounded here, independently of the start:

* **Axis-to-tip radii r_i.** During any motion the TCP velocity is
  v = sum_i qdot_i (a_i x (p_tip - p_i)), so |v| <= sum_i |qdot_i| r_i with
  r_i = the largest distance of the tool tip from the line of axis i over all
  joint angles. r_i is bounded from the URDF and TOOL without sampling:
  the tip seen from axis i is o_{i+1} + R (o_{i+2} + R (... + R t)) with the fixed
  joint offsets o_k (URDF ``origin xyz``, in the frame of axis k-1) and the TOOL
  offset t (in the flange frame). A rotation about axis i does not change the
  distance to it, and the offset o_{i+1} is fixed in the frame of axis i, so
  r_i <= |o_{i+1} perpendicular to a_i| + |o_{i+2}| + ... + |o_6| + |t|
  (triangle inequality; every later offset counted with its full length, i.e. as
  if it could always point straight away from axis i). This holds for every
  joint vector, not only inside the joint limits.
* **Bound at $VEL_AXIS p %.** In a synchronous PTP no axis exceeds p % of its
  maximum speed (URDF velocity, unverified), whatever the start pose, so the TCP
  speed is <= B(p) = sum_i (p / 100) vmax_i r_i. B is linear in p.
* **Case "bound".** If B(1) <= k x max_tcp_speed_mm_s (k = speed_safety_factor), the
  program sets $VEL_AXIS[1..6] = p before the first PTP, p = the largest integer
  >= 1 with B(p) <= k x max_tcp (capped at the configured axis speed).
* **Case "declared start".** Otherwise no start-independent bound exists at an integer
  $VEL_AXIS >= 1 %. The program then assumes the robot stands at ``transit.start_axis``
  (within ``transit.start_tolerance_deg`` on every axis) and $VEL_AXIS for the first
  PTP comes from the transit speed model (kukasim/transit.py), evaluated from the
  declared start and from all 64 corners of the tolerance box; the worst of them must
  be <= k x max_tcp. The header says so, and gives B(1) for reference.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

CASE_BOUND = "bound"
CASE_DECLARED = "declared start"


def axis_tip_radii(robot):
    """r_i (mm, shape (6,)): upper bound of the distance from the line of axis i to the
    tool tip over all joint angles (see the module docstring)."""
    J = robot.joints
    t = robot.T_tool[:3, 3]
    offs = [j.origin_xyz for j in J[1:]] + [t]      # o_2..o_6 (frames of axes 1..5), then TOOL (flange frame)
    r = np.empty(6)
    for i in range(6):
        a = J[i].axis / np.linalg.norm(J[i].axis)
        o = offs[i]
        perp = float(np.linalg.norm(o - np.dot(o, a) * a))
        r[i] = perp + sum(float(np.linalg.norm(v)) for v in offs[i + 1:])
    return r


def tcp_speed_bound(robot, pct):
    """B(pct): start-independent TCP speed bound (mm/s) of a PTP at $VEL_AXIS = pct %."""
    return float(np.sum(robot.vmax * pct / 100.0 * axis_tip_radii(robot)))


def bound_pct(robot, limit_mm_s, pct_max=100):
    """Largest integer p in 1..pct_max with B(p) <= limit, or None if even B(1) > limit."""
    b1 = tcp_speed_bound(robot, 1.0)
    p = min(int(pct_max), math.floor(limit_mm_s / b1 + 1e-12))
    return p if p >= 1 else None


@dataclass
class FirstPTP:
    case: str
    vel_axis_pct: int
    acc_axis_pct: float
    peak_tcp_mm_s: float              # bound case: B(p) (an upper bound); declared: worst model peak
    limit_mm_s: float
    radii_mm: list
    bound_1pct_mm_s: float
    start_axis_deg: list | None = None
    start_tolerance_deg: float | None = None
    peak_nominal_mm_s: float | None = None   # declared: model peak from the nominal start
    target_deg: list | None = None
    notes: list = field(default_factory=list)

    def summary(self):
        d = {"case": self.case, "vel_axis_pct": self.vel_axis_pct, "acc_axis_pct": self.acc_axis_pct,
             "peak_tcp_mm_s": round(self.peak_tcp_mm_s, 1), "limit_mm_s": self.limit_mm_s,
             "axis_tip_radii_mm": [round(v, 1) for v in self.radii_mm],
             "bound_at_1pct_mm_s": round(self.bound_1pct_mm_s, 1)}
        if self.case == CASE_DECLARED:
            d |= {"start_axis_deg": self.start_axis_deg, "start_tolerance_deg": self.start_tolerance_deg,
                  "peak_tcp_nominal_start_mm_s": round(self.peak_nominal_mm_s, 1)}
        return d

    def header(self):
        """KRL comment lines for the program header."""
        r = ", ".join(f"{v:.0f}" for v in self.radii_mm)
        L = [f"; First PTP (from the pose the robot is in when the program starts): case '{self.case}'."]
        if self.case == CASE_BOUND:
            L += [f";   $VEL_AXIS[1..6] = {self.vel_axis_pct} %, $ACC_AXIS[1..6] = {self.acc_axis_pct:g} % before it.",
                  f";   Start-independent bound: TCP speed <= sum_i ($VEL_AXIS % x axis max speed) x r_i = "
                  f"{self.peak_tcp_mm_s:.1f} mm/s",
                  f";   <= {self.limit_mm_s:g} mm/s from ANY start pose (r_i = {r} mm, axis to tool tip;",
                  ";   URDF axis maxima, unverified). Peak TCP speed of the first PTP <= "
                  f"{self.peak_tcp_mm_s:.1f} mm/s (bound)."]
        else:
            s = ", ".join(f"A{i + 1} {v:g}" for i, v in enumerate(self.start_axis_deg))
            L += [f";   No start-independent bound: even at $VEL_AXIS 1 % the bound is {self.bound_1pct_mm_s:.1f} mm/s "
                  f"> {self.limit_mm_s:g} mm/s",
                  f";   (sum_i 1 % x axis max speed x r_i, r_i = {r} mm, axis to tool tip; URDF axis maxima).",
                  f";   THE ROBOT MUST BE AT THE DECLARED START POSE {{{s}}}",
                  f";   (within +-{self.start_tolerance_deg:g} deg on every axis) BEFORE THIS PROGRAM STARTS.",
                  f";   From there: $VEL_AXIS[1..6] = {self.vel_axis_pct} %, $ACC_AXIS[1..6] = {self.acc_axis_pct:g} %; "
                  f"model peak TCP speed of the first PTP {self.peak_tcp_mm_s:.1f} mm/s",
                  f";   (worst over the start and the 64 tolerance-box corners; {self.peak_nominal_mm_s:.1f} mm/s "
                  f"from the start itself) <= {self.limit_mm_s:g} mm/s.",
                  ";   Model estimate (transit speed model), not controller output. From any other pose the speed",
                  ";   is not bounded by this program."]
        return L


class StartPoseError(ValueError):
    """The declared start pose is invalid or its first PTP cannot meet the speed rule."""


def plan_first_ptp(robot, cfg, q_target, acc_pct, pct_max=None):
    """$VEL_AXIS for the first PTP of a program (to joint target q_target, rad).

    acc_pct: the $ACC_AXIS the program sets with it. pct_max: the configured axis speed
    (default transit.vel_axis_pct); the result is never above it."""
    from .transit import TransitPlanner
    tp = cfg.planner["transit"]
    limit = float(tp["speed_safety_factor"]) * float(tp["max_tcp_speed_mm_s"])
    pct_max = int(tp["vel_axis_pct"] if pct_max is None else pct_max)
    radii = axis_tip_radii(robot).tolist()
    b1 = tcp_speed_bound(robot, 1.0)
    p = bound_pct(robot, limit, pct_max)
    if p is not None:
        return FirstPTP(CASE_BOUND, p, acc_pct, tcp_speed_bound(robot, p), limit, radii, b1,
                        target_deg=None if q_target is None else np.degrees(q_target).tolist())
    start = np.asarray(tp["start_axis"], float)
    tol = float(tp["start_tolerance_deg"])
    if start.shape != (6,):
        raise StartPoseError(f"transit.start_axis must be six joint angles, got {tp['start_axis']}")
    lim = np.degrees(robot.limits)
    if ((start - tol < lim[:, 0]) | (start + tol > lim[:, 1])).any():
        raise StartPoseError(f"transit.start_axis {start.tolist()} +- {tol:g} deg is outside the joint limits")
    fp = FirstPTP(CASE_DECLARED, 1, acc_pct, 0.0, limit, radii, b1, start.tolist(), tol, 0.0,
                  None if q_target is None else np.degrees(q_target).tolist())
    if q_target is None:
        return fp
    tpl = TransitPlanner(robot, cfg, None, robot.limits)
    starts = np.radians(np.vstack([start] + [start + tol * np.array(s)
                                             for s in itertools.product((-1.0, 1.0), repeat=6)]))
    step = np.radians(float(tp["ptp_sample_deg"]))

    def peaks(pct):
        out = []
        for q0 in starts:
            n = int(np.ceil(np.abs(q_target - q0).max() / step)) + 1
            q = q0 + np.linspace(0.0, 1.0, max(n, 2))[:, None] * (q_target - q0)
            out.append(tpl.ptp_speed(q, pct, acc_pct)[0])
        return np.array(out)

    pct = pct_max
    pk = peaks(pct)
    while pk.max() > limit and pct > 1:
        pct -= 1
        pk = peaks(pct)
    if pk.max() > limit:
        raise StartPoseError(f"first PTP from the declared start: model peak {pk.max():.0f} mm/s at $VEL_AXIS 1 % "
                             f"is above {limit:g} mm/s")
    fp.vel_axis_pct, fp.peak_tcp_mm_s, fp.peak_nominal_mm_s = pct, float(pk.max()), float(pk[0])
    return fp

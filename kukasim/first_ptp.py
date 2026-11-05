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
* **Case "declared start".** Otherwise B(1) cannot certify any integer $VEL_AXIS >= 1 %.
  B is only an upper bound, so that alone does not show that a start-independent
  setting is impossible; the sampled evidence does: synchronous PTPs at 1 % between
  random poses inside the joint limits (``random_start_peak``) reach more than
  k x max_tcp. The program then assumes the robot stands at ``transit.start_axis``
  (within ``transit.start_tolerance_deg`` on every axis), and $VEL_AXIS for the first
  PTP comes from the transit speed model (kukasim/transit.py), evaluated from a
  sample of starts in the tolerance box (``box_starts``: the 3^6 grid at -tol, 0,
  +tol, i.e. the centre, the 64 corners and the edge / face midpoints, plus
  ``BOX_RANDOM`` uniform interior points with a fixed seed). This is a sample, NOT
  a bound over the box: the leading axis can change inside the box, so the peak
  between samples is not known exactly. The worst sampled peak must therefore stay
  <= (1 - ``transit.start_sample_margin``) x k x max_tcp. The header says so, and
  gives B(1) and the sampled evidence for reference.
* **Case "no first PTP".** A program with no first PTP target (e.g. no IK solution
  for the approach pose) starts with a motion from an unknown pose whose speed is
  not checked; the header says so and the check fails.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

CASE_BOUND = "bound"
CASE_DECLARED = "declared start"
CASE_NONE = "no first PTP"

BOX_LEVELS = (-1.0, 0.0, 1.0)     # grid levels in the tolerance box, x start_tolerance_deg
BOX_RANDOM = 256                  # uniform interior samples on top of the grid
BOX_SEED = 7
EVIDENCE_PAIRS = 3000             # random start / target pairs for the sampled evidence at 1 %
EVIDENCE_SEED = 5


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


def box_starts(start_deg, tol_deg):
    """Sampled start poses (deg, (N,6)) in the tolerance box start +- tol: the nominal start
    first, then the rest of the 3^6 grid (corners, edge and face midpoints), then BOX_RANDOM
    uniform interior points (fixed seed). A sample, not a bound over the box."""
    start = np.asarray(start_deg, float)
    grid = np.array([g for g in itertools.product(BOX_LEVELS, repeat=6) if any(g)])
    rnd = np.random.default_rng(BOX_SEED).uniform(-1.0, 1.0, size=(BOX_RANDOM, 6))
    return start + tol_deg * np.vstack([np.zeros((1, 6)), grid, rnd])


def sync_ptp_peak(robot, q0, q1, pct=100.0, n=400):
    """Sampled peak TCP speed (mm/s) of synchronous PTPs q0 -> q1 ((M,6), rad) at the cruise rate of
    the leading axis at pct % of the URDF axis maxima (every axis at <= its share of its maximum;
    no ramps, so this is the fastest the move can be at that $VEL_AXIS)."""
    dq = q1 - q0
    sdot = np.min(robot.vmax * pct / 100.0 / np.maximum(np.abs(dq), 1e-15), axis=1)     # 1/s
    s = np.linspace(0.0, 1.0, n)
    q = q0[:, None] + s[None, :, None] * dq[:, None]
    tip = (robot.fk_flange(q) @ robot.T_tool)[..., :3, 3]
    dxds = np.linalg.norm(np.diff(tip, axis=1), axis=-1) * (n - 1)
    return (dxds * sdot[:, None]).max(1)


_EVIDENCE = {}


def random_start_peak(robot, pct=1.0):
    """Sampled evidence (mm/s): the largest peak of sync_ptp_peak at pct % over EVIDENCE_PAIRS
    random start / target pairs inside the joint limits (seed EVIDENCE_SEED). A lower estimate of
    the true worst case; if it is above the limit, no start-independent $VEL_AXIS >= pct exists."""
    key = (id(robot), float(pct))
    if key not in _EVIDENCE:
        rng = np.random.default_rng(EVIDENCE_SEED)
        lo, hi = robot.limits[:, 0], robot.limits[:, 1]
        q0 = rng.uniform(lo, hi, size=(EVIDENCE_PAIRS, 6))
        q1 = rng.uniform(lo, hi, size=(EVIDENCE_PAIRS, 6))
        _EVIDENCE[key] = float(sync_ptp_peak(robot, q0, q1, pct).max())
    return _EVIDENCE[key]


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
    peak_tcp_mm_s: float | None       # bound: B(p) (an upper bound); declared: worst sampled model peak; none: None
    limit_mm_s: float
    radii_mm: list
    bound_1pct_mm_s: float
    start_axis_deg: list | None = None
    start_tolerance_deg: float | None = None
    peak_nominal_mm_s: float | None = None   # declared: model peak from the nominal start
    target_deg: list | None = None
    n_starts: int | None = None              # declared: number of sampled starts in the tolerance box
    margin: float | None = None              # declared: required margin of the worst sampled peak
    sampled_1pct_mm_s: float | None = None   # declared: random_start_peak at 1 % (sampled evidence)
    notes: list = field(default_factory=list)

    @property
    def sample_limit_mm_s(self):
        """declared: the worst sampled peak must be <= this (limit reduced by the margin)."""
        return (1.0 - (self.margin or 0.0)) * self.limit_mm_s

    @property
    def ok(self):
        """The first-PTP speed rule is met (bound, or worst sampled peak <= the reduced limit)."""
        if self.case == CASE_NONE:
            return False
        if self.case == CASE_BOUND:
            return self.peak_tcp_mm_s <= self.limit_mm_s
        return self.peak_tcp_mm_s <= self.sample_limit_mm_s + 1e-9

    def summary(self):
        d = {"case": self.case, "vel_axis_pct": self.vel_axis_pct, "acc_axis_pct": self.acc_axis_pct,
             "peak_tcp_mm_s": None if self.peak_tcp_mm_s is None else round(self.peak_tcp_mm_s, 1),
             "peak_kind": {CASE_BOUND: "bound from any start", CASE_NONE: "no first PTP (not checked)",
                           CASE_DECLARED: f"worst sampled over {self.n_starts} starts in the tolerance box "
                                          "(not a bound)"}[self.case],
             "limit_mm_s": self.limit_mm_s, "ok": self.ok,
             "axis_tip_radii_mm": [round(v, 1) for v in self.radii_mm],
             "bound_at_1pct_mm_s": round(self.bound_1pct_mm_s, 1)}
        if self.case == CASE_DECLARED:
            d |= {"sampled_random_start_peak_1pct_mm_s": round(self.sampled_1pct_mm_s, 1),
                  "start_axis_deg": self.start_axis_deg, "start_tolerance_deg": self.start_tolerance_deg,
                  "sampled_starts": self.n_starts, "sample_margin": self.margin,
                  "sample_limit_mm_s": round(self.sample_limit_mm_s, 1),
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
        elif self.case == CASE_NONE:
            L += [";   NO FIRST PTP: the program has no first PTP target. Its first motion starts from an UNKNOWN",
                  ";   pose (the pose the robot is in) and its speed is NOT checked (no bound, no model peak).",
                  f";   $VEL_AXIS[1..6] = {self.vel_axis_pct} %, $ACC_AXIS[1..6] = {self.acc_axis_pct:g} % are set "
                  "but do not limit a LIN."]
        else:
            s = ", ".join(f"A{i + 1} {v:g}" for i, v in enumerate(self.start_axis_deg))
            L += [f";   No start-independent $VEL_AXIS: sampled synchronous PTPs at $VEL_AXIS 1 % between random poses",
                  f";   inside the joint limits reach {self.sampled_1pct_mm_s:.1f} mm/s > {self.limit_mm_s:g} mm/s "
                  f"(sampled evidence; the upper bound",
                  f";   sum_i 1 % x axis max speed x r_i is {self.bound_1pct_mm_s:.1f} mm/s, r_i = {r} mm, "
                  "axis to tool tip; URDF axis maxima).",
                  f";   THE ROBOT MUST BE AT THE DECLARED START POSE {{{s}}}",
                  f";   (within +-{self.start_tolerance_deg:g} deg on every axis) BEFORE THIS PROGRAM STARTS.",
                  f";   From there: $VEL_AXIS[1..6] = {self.vel_axis_pct} %, $ACC_AXIS[1..6] = {self.acc_axis_pct:g} %; "
                  f"model peak TCP speed of the first PTP {self.peak_tcp_mm_s:.1f} mm/s",
                  f";   (worst sampled over {self.n_starts} starts in the tolerance box, NOT a bound; "
                  f"{self.peak_nominal_mm_s:.1f} mm/s from the start itself)",
                  f";   <= {self.sample_limit_mm_s:g} mm/s = (1 - {self.margin:g} margin for unsampled starts) x "
                  f"{self.limit_mm_s:g} mm/s.",
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
    if q_target is None:
        # no first PTP: the first motion starts from an unknown pose; nothing is checked
        return FirstPTP(CASE_NONE, 1, acc_pct, None, limit, radii, b1)
    p = bound_pct(robot, limit, pct_max)
    if p is not None:
        return FirstPTP(CASE_BOUND, p, acc_pct, tcp_speed_bound(robot, p), limit, radii, b1,
                        target_deg=np.degrees(q_target).tolist())
    start = np.asarray(tp["start_axis"], float)
    tol = float(tp["start_tolerance_deg"])
    if start.shape != (6,):
        raise StartPoseError(f"transit.start_axis must be six joint angles, got {tp['start_axis']}")
    lim = np.degrees(robot.limits)
    if ((start - tol < lim[:, 0]) | (start + tol > lim[:, 1])).any():
        raise StartPoseError(f"transit.start_axis {start.tolist()} +- {tol:g} deg is outside the joint limits")
    margin = float(tp["start_sample_margin"])
    if not 0.0 <= margin < 1.0:
        raise StartPoseError(f"transit.start_sample_margin must be in [0, 1), got {margin}")
    fp = FirstPTP(CASE_DECLARED, 1, acc_pct, 0.0, limit, radii, b1, start.tolist(), tol, 0.0,
                  np.degrees(q_target).tolist(), margin=margin, sampled_1pct_mm_s=random_start_peak(robot, 1.0))
    tpl = TransitPlanner(robot, cfg, None, robot.limits)
    starts = np.radians(box_starts(start, tol))
    fp.n_starts = len(starts)
    step = np.radians(float(tp["ptp_sample_deg"]))
    paths = []
    for q0 in starts:
        n = int(np.ceil(np.abs(q_target - q0).max() / step)) + 1
        paths.append(q0 + np.linspace(0.0, 1.0, max(n, 2))[:, None] * (q_target - q0))

    def peaks(pct):
        return np.array([tpl.ptp_speed(q, pct, acc_pct)[0] for q in paths])

    lim_s = fp.sample_limit_mm_s
    pct = pct_max
    pk = peaks(pct)
    while pk.max() > lim_s and pct > 1:
        pct -= 1
        pk = peaks(pct)
    if pk.max() > lim_s:
        raise StartPoseError(f"first PTP from the declared start: worst sampled model peak {pk.max():.0f} mm/s at "
                             f"$VEL_AXIS 1 % is above {lim_s:g} mm/s ({1 - margin:g} x {limit:g} mm/s)")
    fp.vel_axis_pct, fp.peak_tcp_mm_s, fp.peak_nominal_mm_s = pct, float(pk.max()), float(pk[0])
    return fp

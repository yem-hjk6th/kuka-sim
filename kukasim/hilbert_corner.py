"""Corner test: one Hilbert curve layer with a different corner speed ratio at every corner.

Simulation only. The lab prints one layer of a Hilbert curve (many 90 deg
corners) with the extruder rpm held fixed. The TCP speed around each corner is
r x the straight speed, where r differs per corner. Bead width scales roughly
with flow / speed, so r < 1 should widen the corner bead and r > 1 thin it; one
print then shows the trend of corner bead width over r.

Two ways to set the corner speed (``speed_mode``):
  * ``krl`` (default): the program itself splits each leg at +-``d_zone_mm``
    around every corner and sets ``$VEL.CP = r x v`` on the two corner
    sub-segments, back to v after them. The ratio is path-synchronous in the
    program (no PC in the loop); d_zone must be >= 2 x C_DIS so the corner
    blend fits inside the zone.
  * ``pc``: the program runs at one $VEL.CP and a PC-side controller switches
    ``$OV_PRO`` to ``r x ov_straight`` from ``window_before_mm`` before to
    ``window_after_mm`` after each vertex, per schedule.csv. This needs a
    PC-side schedule follower, which this repository does NOT provide.

The layer is planar or tilted: z(x, y) = z0 + kx (x - x0) + ky (y - y0)
(``layer_z``; default flat at bed top_z + layer height).

Obstacles are those in config/cell.yaml (clamps and the optional ``keep_out``
list, empty in the repository). "Checks passed" therefore means checked against
cell.yaml only: site keep-outs must be checked locally before running.

This module builds the path, the per-corner schedule, a simple corner-speed
model (a model estimate, to be compared with the speed the controller logs)
and checks over the whole path with the existing kinematics, DP planner and
collision model. Parameters live in ``config/planner.yaml`` (``hilbert_corner``).

Corner-speed model (stated assumptions, none verified on the controller):
  * KRC approximate positioning with C_DIS replaces each corner by a curve that
    starts and ends at d = min(C_DIS, half the shorter adjacent segment) from
    the vertex; it is modelled as the quadratic Bezier with the vertex as
    control point (minimum radius d / sqrt(2) at a 90 deg corner).
  * The commanded path speed is $VEL.CP * override. The corner value applies
    from ``zone_before`` before the vertex to ``zone_after`` after it (krl:
    +-d_zone, no delay; pc: the override windows, each switch delayed by
    ``pc_latency``).
  * The speed follows the command with a trapezoid profile (constant path
    acceleration ``path_acc``, no jerk limit), is capped in the blend by
    sqrt(lateral_acc * R), and is zero at the exact stops at start and end.
  * ``corner_dip_factor`` scales the commanded speed inside each blend (default
    1.0 = no dip). Real controllers slow down in blended 90 deg corners below
    the programmed speed; the factor is a placeholder for that, the logged
    speed decides. Predicted bead widths are model-only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .collision import keep_out_boxes
from .first_ptp import StartPoseError, plan_first_ptp
from .krl import NOT_RUN, fang, fnum, krl_name, lin_pose, ptp_axes
from .planner import Planner, tcp_poses
from .rotations import matrix_to_abc
from .toolpaths import Toolpath

DIRS4 = {(1, 0): "+X", (-1, 0): "-X", (0, 1): "+Y", (0, -1): "-Y"}


# ---------------------------------------------------------------- parameters
def corner_params(cfg, **overrides):
    """hilbert_corner section of planner.yaml with keyword overrides (None = keep)."""
    p = copy.deepcopy(cfg.planner["hilbert_corner"])
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"unknown hilbert_corner parameter {k!r}")
        if v is not None:
            p[k] = v
    if p["straight_speed_mm_s"] is None:
        p["straight_speed_mm_s"] = cfg.print_speed
    p["ratios"] = [float(r) for r in p["ratios"]]
    if p["speed_mode"] not in ("krl", "pc"):
        raise ValueError(f"hilbert_corner.speed_mode must be 'krl' or 'pc', not {p['speed_mode']!r}")
    if p["speed_mode"] == "krl":
        d = float(p["d_zone_mm"])
        if d < 2.0 * float(p["c_dis_mm"]):
            raise ValueError(f"hilbert_corner.d_zone_mm = {d:g} mm must be >= 2 x c_dis_mm "
                             f"(= {2 * float(p['c_dis_mm']):g} mm) so the corner blend fits inside the zone")
        p["zone_before_mm"] = p["zone_after_mm"] = d
        p["latency_s"] = 0.0
    else:
        p["zone_before_mm"] = float(p["window_before_mm"])
        p["zone_after_mm"] = float(p["window_after_mm"])
        p["latency_s"] = float(p["pc_latency_s"])
    lz = {"z0": None, "kx": 0.0, "ky": 0.0, "x0": None, "y0": None} | dict(p.get("layer_z") or {})
    p["layer_z"] = lz
    return p


def _zones(p):
    """(before, after) corner zone lengths: the effective zones if set, else the PC override windows."""
    if "zone_before_mm" in p:
        return float(p["zone_before_mm"]), float(p["zone_after_mm"])
    return float(p["window_before_mm"]), float(p["window_after_mm"])


def layer_z_law(p, cfg, center):
    """(z0, kx, ky, x0, y0) of z(x, y) = z0 + kx (x - x0) + ky (y - y0); defaults: flat at
    bed top_z + layer height, reference point = centre of the square."""
    lz = p["layer_z"]
    z0 = cfg.bed()["top_z"] + cfg.layer_height if lz["z0"] is None else float(lz["z0"])
    x0 = center[0] if lz["x0"] is None else float(lz["x0"])
    y0 = center[1] if lz["y0"] is None else float(lz["y0"])
    return z0, float(lz["kx"]), float(lz["ky"]), x0, y0


def apply_layer_z(points, law):
    z0, kx, ky, x0, y0 = law
    out = np.array(points, float)
    out[:, 2] = z0 + kx * (out[:, 0] - x0) + ky * (out[:, 1] - y0)
    return out


# ------------------------------------------------------------------ geometry
def hilbert_grid(order):
    """Integer vertices (4^n, 2) of the Hilbert curve on the 2^n x 2^n grid,
    from (0, 0) to (2^n - 1, 0), every step one grid unit."""
    if order < 1:
        raise ValueError("order >= 1")
    n = 1 << order
    out = np.zeros((n * n, 2), int)
    for d in range(n * n):
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        out[d] = (x, y)
    return out


def hilbert_path(order, side, center, z):
    """(4^n, 3) vertices in mm: the curve spans a side x side square around center."""
    g = hilbert_grid(order).astype(float)
    step = side / ((1 << order) - 1)
    xy = (g - g.max(0) / 2.0) * step + np.asarray(center, float)
    return np.column_stack([xy, np.full(len(xy), float(z))])


def find_corners(points, tol=1e-9):
    """Indices of vertices where the direction changes, and the turn (+1 left, -1 right)."""
    d = np.diff(points[:, :2], axis=0)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    idx = np.nonzero(np.abs(cross) > tol)[0] + 1
    return idx, np.sign(cross[idx - 1]).astype(int)


# --------------------------------------------------------------- reachability
def single_angle_planner(robot, cfg, theta_deg):
    pl = Planner(robot, cfg)
    pl.thetas = np.radians(np.array([float(theta_deg)]))
    return pl


def reach_map(planner, cfg, z, grid_mm):
    """Feasibility of the vertical nozzle at height z on a grid over the bed outline
    (some IK slot inside the limits with margin and collision-free)."""
    bed = cfg.bed()
    xs = np.arange(bed["x"][0], bed["x"][1] + 1e-9, grid_mm)
    ys = np.arange(bed["y"][0], bed["y"][1] + 1e-9, grid_mm)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    D = np.tile([0, 0, -1.0], (len(P), 1))
    T = tcp_poses(P, D, planner.thetas).reshape(-1, 4, 4)
    Q, ok = planner.robot.ik(T, limits=planner.hard_limits)
    ok &= np.abs(Q[..., 4]) >= np.radians(planner.p["min_abs_a5_deg"])
    feas = np.zeros(len(P), bool)
    i, k = np.nonzero(ok)
    if len(i):
        c = planner.coll.clearance(Q[i, k])
        np.logical_or.at(feas, i, c >= 0)
    return xs, ys, feas.reshape(X.shape)


def box_xy_distance(X, Y, boxes):
    """XY distance of the points (X, Y) (arrays of one shape) to the nearest box footprint (inf if none)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    d = np.full(X.shape, np.inf)
    for b in boxes:
        dx = np.maximum(np.maximum(b.lo[0] - X, X - b.hi[0]), 0.0)
        dy = np.maximum(np.maximum(b.lo[1] - Y, Y - b.hi[1]), 0.0)
        d = np.minimum(d, np.hypot(dx, dy))
    return d


def _signed_box_distance(P, lo, hi):
    """Signed distance of points (N,3) to an AABB: > 0 outside, < 0 inside (depth)."""
    d = np.maximum(lo - P, P - hi)
    out = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    return np.where((d <= 0).all(1), d.max(1), out)


def fit_square(xs, ys, feas, margin_mm, round_mm=10.0, prefer_xy=None):
    """Largest axis-aligned square of feasible grid points, at least margin_mm away
    from every infeasible point and from the grid border. Ties: closest to prefer_xy."""
    from scipy.ndimage import binary_erosion
    res = xs[1] - xs[0]
    m = int(np.ceil(margin_mm / res))
    ok = binary_erosion(feas, structure=np.ones((2 * m + 1, 2 * m + 1), bool), border_value=0) \
        if m > 0 else feas.copy()
    # classic DP: S[i, j] = side (in points) of the largest square with corner (i, j) at its top right
    S = np.zeros(ok.shape, int)
    for i in range(ok.shape[0]):
        for j in range(ok.shape[1]):
            if ok[i, j]:
                S[i, j] = 1 if i == 0 or j == 0 else 1 + min(S[i - 1, j], S[i, j - 1], S[i - 1, j - 1])
    kmax = int(S.max())
    if kmax < 2:
        raise ValueError("no feasible square on the bed")
    side = np.floor((kmax - 1) * res / round_mm) * round_mm
    ii, jj = np.nonzero(S == kmax)
    cx = xs[jj] - (kmax - 1) * res / 2
    cy = ys[ii] - (kmax - 1) * res / 2
    if prefer_xy is None:
        prefer_xy = (0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1]))
    best = int(np.argmin(np.hypot(cx - prefer_xy[0], cy - prefer_xy[1])))
    return float(side), (float(cx[best]), float(cy[best]))


# ------------------------------------------------------------------ schedule
def corner_windows(seg, kinds, p):
    """Straight length at the straight override before / after each waypoint.

    seg[j] is the length from waypoint j to j+1; kinds[j] is 'start', 'corner'
    or 'end'. The straight part of a segment is its length minus the override
    windows at both ends (start / end points have no window)."""
    wb, wa = _zones(p)
    after = np.array([wa if k == "corner" else 0.0 for k in kinds])
    before = np.array([wb if k == "corner" else 0.0 for k in kinds])
    straight = seg - after[:-1] - before[1:]
    s_in = np.r_[np.nan, straight]
    s_out = np.r_[straight, np.nan]
    return s_in, s_out


def too_close(straight_in, straight_out, kinds, min_straight):
    """Corners whose straight before or after (between corner zones) is shorter than min_straight."""
    is_c = np.array([k == "corner" for k in kinds])
    return is_c & ((np.nan_to_num(straight_in, nan=np.inf) < min_straight) |
                   (np.nan_to_num(straight_out, nan=np.inf) < min_straight))


def assign_ratios(n_corners, eligible, ratios, min_repeats, seed):
    """Seeded random assignment: every ratio at least min_repeats times (if enough
    eligible corners), the remainder spread over distinct ratios; r = 1 elsewhere."""
    rng = np.random.default_rng(seed)
    E = np.nonzero(eligible)[0]
    ratios = list(ratios)
    reps = len(E) // len(ratios)
    pool = ratios * reps
    rem = len(E) - len(pool)
    if rem:
        pool += list(rng.choice(ratios, size=rem, replace=False))
    pool = np.array(pool, float)
    rng.shuffle(pool)
    r = np.ones(n_corners)
    r[E] = pool
    return r


# --------------------------------------------------------- blended path model
def _bezier(P0, V, P2, t):
    t = np.asarray(t, float)[:, None]
    return (1 - t) ** 2 * P0 + 2 * t * (1 - t) * V + t * t * P2


def _bezier_radius(P0, V, P2, t):
    t = np.asarray(t, float)[:, None]
    d1 = 2 * (1 - t) * (V - P0) + 2 * t * (P2 - V)
    d2 = np.broadcast_to(2 * (P2 - 2 * V + P0), d1.shape)
    cr = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    n = np.linalg.norm(d1[:, :2], axis=1)
    return np.where(cr > 1e-12, n ** 3 / np.maximum(cr, 1e-300), np.inf)


def blended_path(W, d_eff, ds):
    """Dense samples of the programmed polyline W (waypoints) with every interior
    vertex j replaced by the Bezier blend of half-length d_eff[j].

    Returns points (M,3), s_prog (M,) position along the programmed polyline,
    radius (M,) curvature radius (inf on straights), seg (M,) waypoint segment
    index the sample belongs to (blend samples belong to the corner's waypoint
    via ``vert`` (M,), -1 elsewhere)."""
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    s_w = np.r_[0.0, np.cumsum(seg)]
    total = s_w[-1]
    n = int(np.ceil(total / ds)) + 1
    s = np.linspace(0, total, n)
    # make sure every vertex and blend end is sampled exactly
    extra = [s_w]
    for j in range(1, len(W) - 1):
        extra.append([s_w[j] - d_eff[j], s_w[j] + d_eff[j]])
    req = np.unique(np.clip(np.concatenate([np.asarray(e, float) for e in extra]), 0, total))
    req = req[np.r_[True, np.diff(req) > 1e-6]]
    near = np.abs(s[:, None] - req[None, :]).min(1) < 0.25 * ds
    s = np.sort(np.concatenate([s[~near], req]))
    j_seg = np.clip(np.searchsorted(s_w, s, side="right") - 1, 0, len(seg) - 1)
    u = (s - s_w[j_seg]) / seg[j_seg]
    P = W[j_seg] + u[:, None] * (W[j_seg + 1] - W[j_seg])
    R = np.full(len(s), np.inf)
    vert = np.full(len(s), -1)
    for j in range(1, len(W) - 1):
        d = d_eff[j]
        sel = (s >= s_w[j] - d - 1e-9) & (s <= s_w[j] + d + 1e-9)
        if not sel.any() or d <= 0:
            continue
        uin = (W[j] - W[j - 1]) / seg[j - 1]
        uout = (W[j + 1] - W[j]) / seg[j]
        P0, P2 = W[j] - d * uin, W[j] + d * uout
        t = np.clip((s[sel] - (s_w[j] - d)) / (2 * d), 0, 1)
        P[sel] = _bezier(P0, W[j], P2, t)
        R[sel] = _bezier_radius(P0, W[j], P2, t)
        vert[sel] = j
    return P, s, R, vert, s_w


def speed_profile(s_path, v_cmd, acc):
    """Trapezoid profile: forward / backward passes with v^2 <= v0^2 + 2 a ds, v = 0 at both ends."""
    v = np.minimum(v_cmd, np.inf).astype(float)
    v[0] = v[-1] = 0.0
    ds = np.diff(s_path)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * acc * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * acc * ds[i]))
    return v


def time_along(s_path, v):
    ds = np.diff(s_path)
    vm = 0.5 * (v[1:] + v[:-1])
    dt = np.where(vm > 1e-9, ds / np.maximum(vm, 1e-9), 0.0)
    # exact stops: time to accelerate from 0 over the first interval is finite (vm > 0)
    return np.r_[0.0, np.cumsum(dt)]


# --------------------------------------------------------------------- result
@dataclass
class CornerTest:
    params: dict
    side: float
    center: tuple
    z: float
    points: np.ndarray          # (4^n, 3) Hilbert vertices
    W: np.ndarray               # (K+2, 3) waypoints: start, corners, end
    kinds: list
    vertex_index: np.ndarray    # waypoint -> Hilbert vertex index
    turn: np.ndarray            # (K+2,) +1 left, -1 right, 0 start/end
    seg: np.ndarray             # (K+1,) waypoint segment lengths
    s_w: np.ndarray             # (K+2,) programmed arc length at each waypoint
    straight_in: np.ndarray
    straight_out: np.ndarray
    excluded: np.ndarray        # (K+2,) bool: corner too close to a neighbour -> r = 1
    r_cmd: np.ndarray           # (K+2,) commanded ratio (1 at start/end)
    ov_corner: np.ndarray       # (K+2,) corner override % (integer)
    r_eff: np.ndarray           # ov_corner / ov_straight
    d_eff: np.ndarray           # effective blend half-length
    r_blend: np.ndarray         # minimum blend radius
    v_corner: np.ndarray        # predicted speed at the vertex (model)
    v_straight: np.ndarray      # predicted speed on the incoming straight (model)
    ratio_pred: np.ndarray      # v_corner / v_straight (model)
    windows: np.ndarray         # (K+2, 2) programmed s of the override switch (before, after)
    dense: dict = field(default_factory=dict)   # dense path, speed profile, joints
    checks: list = field(default_factory=list)
    reach: tuple = None         # (xs, ys, feas) when the square was fitted
    plan: object = None
    z_law: tuple = None         # (z0, kx, ky, x0, y0)
    approach_q: np.ndarray = None   # joint target (rad) of the approach PTP, from IK with the hard limits

    @property
    def corner_ids(self):
        return np.array([i - 1 if k == "corner" else -1 for i, k in enumerate(self.kinds)])

    @property
    def ok(self):
        return all(c["ok"] for c in self.checks if c["severity"] == "error")

    @property
    def mode(self):
        return self.params["speed_mode"]

    @property
    def v_program(self):
        """$VEL.CP (mm/s) on the straights at 100 % override."""
        if self.mode == "krl":
            return self.params["straight_speed_mm_s"]
        return self.params["straight_speed_mm_s"] * 100.0 / self.params["straight_override_pct"]

    @property
    def v_corner_cmd(self):
        """Commanded corner speed (mm/s) at the straight override (krl: the corner $VEL.CP)."""
        return self.params["straight_speed_mm_s"] * self.r_eff


def validate_geometry(W, seg, kinds, p, d_eff_cap=None):
    """Guards (fail loudly): every waypoint segment >= 2 x C_DIS, corner zones disjoint, every
    LIN of the export longer than min_lin_mm. Raises ValueError with the offending segments."""
    c2 = 2.0 * float(p["c_dis_mm"])
    min_lin = float(p["min_lin_mm"])
    errs = []
    bad = np.nonzero(seg < c2 - 1e-9)[0]
    if len(bad):
        errs.append(f"segment(s) {bad.tolist()} shorter than 2 x C_DIS = {c2:g} mm (min {seg.min():.2f} mm)")
    wb, wa = _zones(p)
    after = np.array([wa if k == "corner" else 0.0 for k in kinds])
    before = np.array([wb if k == "corner" else 0.0 for k in kinds])
    straight = seg - after[:-1] - before[1:]
    bad = np.nonzero(straight <= 0)[0]
    if len(bad):
        errs.append(f"corner zones overlap on segment(s) {bad.tolist()} ({wb:g} mm before + {wa:g} mm after "
                    f"the vertices > segment length, min {seg.min():.2f} mm)")
    if p["speed_mode"] == "krl":
        # LINs: the straight between zones and the two zone halves of every corner
        lins = np.r_[straight, before[before > 0], after[after > 0]]
    else:
        lins = seg
    if len(lins) and lins.min() <= min_lin:
        errs.append(f"a LIN would be {lins.min():.2f} mm long (must be > {min_lin:g} mm)")
    if float(p["approach_lift_mm"]) <= min_lin:
        errs.append(f"approach_lift_mm {p['approach_lift_mm']:g} must be > {min_lin:g} mm")
    if errs:
        raise ValueError("Hilbert corner test geometry rejected: " + "; ".join(errs)
                         + ". Use a larger side, a lower order, a smaller C_DIS or smaller corner zones.")
    return straight


def generate(cfg, robot, **overrides):
    """Build the corner test and run every check. Keyword arguments override
    entries of the hilbert_corner section (see config/planner.yaml).
    Raises ValueError for a geometry the export cannot represent (validate_geometry)."""
    p = corner_params(cfg, **overrides)
    bed = cfg.bed()
    pl = single_angle_planner(robot, cfg, p["theta_deg"])
    reach = None
    side, center = p["side_mm"], p["center_xy"]
    if side is None or center is None:
        z_fit = layer_z_law(p, cfg, (0.0, 0.0))[0]
        xs, ys, feas = reach_map(pl, cfg, z_fit, float(p["fit_grid_mm"]))
        # keep a clear margin from the clamp boxes (and any keep-out) in cell.yaml
        feas = feas & (box_xy_distance(*np.meshgrid(xs, ys), pl.coll.boxes[1:]) >= float(p["clamp_margin_mm"]))
        reach = (xs, ys, feas)
        fs, fc = fit_square(*reach, float(p["fit_margin_mm"]))
        if p["max_side_mm"] is not None:
            fs = min(fs, float(p["max_side_mm"]))
        side = fs if side is None else float(side)
        center = fc if center is None else tuple(center)
    side, center = float(side), (float(center[0]), float(center[1]))
    if side <= 0:
        raise ValueError(f"Hilbert corner test: side {side:g} mm must be > 0")
    law = layer_z_law(p, cfg, center)

    pts = apply_layer_z(hilbert_path(int(p["order"]), side, center, law[0]), law)
    cidx, turns = find_corners(pts)
    vi = np.r_[0, cidx, len(pts) - 1]
    W = pts[vi]
    kinds = ["start"] + ["corner"] * len(cidx) + ["end"]
    turn = np.r_[0, turns, 0]
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    validate_geometry(W, seg, kinds, p)
    s_in, s_out = corner_windows(seg, kinds, p)
    is_c = np.array([k == "corner" for k in kinds])
    excluded = too_close(s_in, s_out, kinds, float(p["min_straight_mm"]))
    r_c = assign_ratios(int(is_c.sum()), ~excluded[is_c], p["ratios"], int(p["min_repeats"]), p["seed"])
    r_cmd = np.ones(len(W))
    r_cmd[is_c] = r_c
    v_s = float(p["straight_speed_mm_s"])
    if p["speed_mode"] == "krl":
        # $VEL.CP is written with 0.01 mm/s resolution; the override stays at 100 %
        ov_s = 100.0
        r_eff = np.where(is_c, np.round(v_s * r_cmd / 1000.0, 5) * 1000.0 / v_s, 1.0)
        ov_c = 100.0 * r_eff                                   # equivalent override (model only)
    else:
        ov_s = float(p["straight_override_pct"])
        ov_c = np.where(is_c, np.round(r_cmd * ov_s), ov_s)
        r_eff = ov_c / ov_s
    zb, za = _zones(p)

    # effective blend: KRC limits the approximation distance to half the shorter programmed segment
    # (krl mode: the corner sub-segments are d_zone long)
    d_eff = np.zeros(len(W))
    adj = np.minimum(seg[:-1], seg[1:]) if p["speed_mode"] == "pc" else np.full(len(W) - 2, min(zb, za))
    d_eff[1:-1] = np.minimum(float(p["c_dis_mm"]), 0.5 * adj)
    r_blend = np.where(is_c, d_eff / np.sqrt(2.0), np.inf)   # 90 deg corner; exact value from the dense R below

    P, s_prog, R, vert, s_w = blended_path(W, d_eff, float(p["ds_mm"]))
    s_path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    v_prog = v_s * 100.0 / ov_s
    lat = float(p["latency_s"])
    ov = np.full(len(s_prog), ov_s)
    windows = np.full((len(W), 2), np.nan)
    for j in np.nonzero(is_c)[0]:
        a = s_w[j] - zb
        b = s_w[j] + za
        windows[j] = (a, b)
        # pc mode: the new override takes effect pc_latency later (distance at the speed before the switch)
        a_eff = a + lat * v_s
        b_eff = b + lat * v_prog * ov_c[j] / 100.0
        ov[(s_prog >= a_eff) & (s_prog < b_eff)] = ov_c[j]
    v_cmd = v_prog * ov / 100.0
    dip = float(p["corner_dip_factor"])
    if not 0 < dip <= 1.5:
        raise ValueError(f"hilbert_corner.corner_dip_factor {dip:g} must be in (0, 1.5]")
    v_cmd = np.where(vert >= 0, v_cmd * dip, v_cmd)
    v_lat = np.sqrt(float(p["lateral_acc_mm_s2"]) * R)
    v = speed_profile(s_path, np.minimum(v_cmd, v_lat), float(p["path_acc_mm_s2"]))
    t = time_along(s_path, v)

    v_corner = np.full(len(W), np.nan)
    v_str = np.full(len(W), np.nan)
    for j in range(len(W)):
        k = int(np.argmin(np.abs(s_prog - s_w[j])))
        v_corner[j] = v[k]
        if j > 0:
            a = s_w[j - 1] + (za if kinds[j - 1] == "corner" else 0.0)
            b = s_w[j] - (zb if kinds[j] == "corner" else 0.0)
            km = int(np.argmin(np.abs(s_prog - 0.5 * (a + b))))
            v_str[j] = v[km]
        if is_c[j]:
            r_blend[j] = float(R[vert == j].min()) if (vert == j).any() else np.inf
    if np.any(v_str[is_c] <= 1e-9):
        bad = (np.nonzero(is_c & (v_str <= 1e-9))[0] - 1).tolist()
        raise ValueError(f"Hilbert corner test: zero model speed on the straight before corner(s) {bad}; "
                         "the straights are too short for the speed profile")
    ratio_pred = np.where(is_c, v_corner / np.where(is_c, v_str, 1.0), np.nan)

    ct = CornerTest(p, side, center, law[0], pts, W, kinds, vi, turn, seg, s_w, s_in, s_out, excluded,
                    r_cmd, ov_c, r_eff, d_eff, r_blend, v_corner, v_str, ratio_pred, windows,
                    dense={"P": P, "s_prog": s_prog, "s_path": s_path, "R": R, "v": v, "v_cmd": v_cmd,
                           "t": t, "vert": vert},
                    reach=reach, z_law=law)
    run_checks(ct, cfg, robot, pl)
    return ct


# --------------------------------------------------------------------- checks
def _check(name, ok, value, limit, detail="", severity="error"):
    return {"name": name, "ok": bool(ok), "value": value, "limit": limit, "detail": detail,
            "severity": severity}


def run_checks(ct, cfg, robot, pl):
    p = ct.params
    P = ct.dense["P"]
    N = len(P)
    dirs = np.tile([0, 0, -1.0], (N, 1))
    tp = Toolpath("hilbert_corner", P, dirs, np.zeros(N), np.zeros(N), ct.dense["s_path"], 0.0)
    res = pl.plan(tp)
    ct.plan = res
    ct.dense["q"] = res.q
    f = res.feasible
    checks = []
    checks.append(_check(
        "reachable, joint limits (with margin) and collision-free in one IK configuration",
        res.complete, f"{100 * f.mean():.1f} % planned, {len(res.segments)} segment(s)", "100 %, 1 segment",
        "" if res.complete else f"no feasible state at points {res.no_state}; breaks at {res.breaks}"))
    q = res.q[f]
    if len(q):
        lm = float(np.nanmin(res.limit_margin))
        checks.append(_check("joint-limit margin beyond the configured margin", lm >= 0, f"{lm:.1f} deg",
                             f">= 0 (margin {cfg.limit_margin_deg:g} deg)"))
        a5 = float(np.abs(np.degrees(q[:, 4])).min())
        checks.append(_check("wrist singularity: min abs(A5)", a5 >= p["min_abs_a5_deg"], f"{a5:.1f} deg",
                             f">= {p['min_abs_a5_deg']} deg"))
        elb = float(np.abs(np.degrees(q[:, 2] - robot.delta3)).min())
        checks.append(_check("elbow singularity: min abs(A3 - stretched-arm angle)", elb >= p["min_elbow_deg"],
                             f"{elb:.1f} deg", f">= {p['min_elbow_deg']} deg"))
        wc = robot.wrist_center(robot.fk_flange(q))
        sh = float(np.hypot(wc[:, 0], wc[:, 1]).min())
        checks.append(_check("shoulder singularity: min wrist-centre distance to the A1 axis",
                             sh >= p["min_shoulder_mm"], f"{sh:.0f} mm", f">= {p['min_shoulder_mm']} mm"))
        clr, det = pl.coll.clearance(q, return_detail=True)
        worst = min(det, key=lambda k: det[k].min())
        checks.append(_check("collision: min clearance to bed (with gap) / clamps / floor (cell.yaml)",
                             clr.min() >= 0, f"{clr.min():.1f} mm ({worst})", ">= 0 mm"))
        # joint speed at the highest commanded TCP speed anywhere on the path
        v_hi = p["straight_speed_mm_s"] * max(1.0, ct.r_eff.max())
        dq = np.abs(np.diff(res.q, axis=0)) / np.diff(ct.dense["s_path"])[:, None] * v_hi
        frac = float(np.nanmax(dq / (robot.vmax * pl.p["velocity_scale"])))
        checks.append(_check(f"joint speed at {v_hi:.0f} mm/s everywhere (fraction of URDF velocity)",
                             frac <= 1.0, f"{frac:.3f}", "<= 1 (URDF velocities unverified)"))
    # keep-out zones: the robot / tool on the planned states and the nozzle path itself (points
    # inside a zone have no feasible state, so the path is checked directly as well)
    ko = keep_out_boxes(cfg)
    kmin = np.inf
    for b in ko:
        kmin = min(kmin, float(_signed_box_distance(P, b.lo, b.hi).min()) - b.required_clearance)
        if len(q):
            kmin = min(kmin, float(det[b.name].min()))
    checks.append(_check("keep-out zones from cell.yaml keep_out (site keep-outs: check locally)",
                         kmin >= 0, f"{len(ko)} zone(s)" + (f", min clearance {kmin:.1f} mm" if ko else
                                                             " defined (none committed)"), ">= 0 mm"))
    bed = cfg.bed()
    m = float(p["bed_edge_margin_mm"])
    inside = (P[:, 0] >= bed["x"][0] + m) & (P[:, 0] <= bed["x"][1] - m) & \
             (P[:, 1] >= bed["y"][0] + m) & (P[:, 1] <= bed["y"][1] - m)
    dmin = float(np.min(np.minimum.reduce([P[:, 0] - bed["x"][0], bed["x"][1] - P[:, 0],
                                           P[:, 1] - bed["y"][0], bed["y"][1] - P[:, 1]])))
    checks.append(_check("nozzle over the bed outline", inside.all(), f"{dmin:.1f} mm to the nearest edge",
                         f">= {m:g} mm"))
    gap_nom = float(P[:, 2].min() - bed["top_z"])
    gap_hi = float(P[:, 2].min() - p["bed_top_max_z"])
    checks.append(_check("min nozzle-to-bed distance (vs highest measured bed point)",
                         gap_hi >= p["min_nozzle_bed_mm"],
                         f"{gap_hi:.2f} mm ({gap_nom:.2f} mm vs top_z {bed['top_z']})",
                         f">= {p['min_nozzle_bed_mm']} mm vs z {p['bed_top_max_z']}"))
    cm = float(p["clamp_margin_mm"])
    dxy = float(box_xy_distance(P[:, 0], P[:, 1], pl.coll.boxes[1:]).min())
    checks.append(_check("XY margin of the path to the clamp boxes (inflated, cell.yaml)", dxy >= cm,
                         f"{dxy:.1f} mm", f">= {cm:g} mm"))
    # approach / retract points: IK with the hard limits (margin), collision-free, the slot
    # closest to the planned start (the approach PTP target) and to the planned end
    lift = np.array([0, 0, float(p["approach_lift_mm"])])
    ends = np.array([ct.W[0] + lift, ct.W[-1] + lift])
    T = tcp_poses(ends, np.tile([0, 0, -1.0], (2, 1)), pl.thetas).reshape(-1, 4, 4)
    Q, okk = robot.ik(T, limits=pl.hard_limits)
    end_ok = []
    ref = [res.q[0], res.q[-1]]
    for i in range(2):
        k = np.nonzero(okk[i])[0]
        if len(k):
            k = k[pl.coll.clearance(Q[i, k]) >= 0]
        end_ok.append(bool(len(k)))
        if i == 0 and len(k) and not np.isnan(ref[0][0]):
            ct.approach_q = Q[0, k[np.argmin(np.abs(Q[0, k] - ref[0]).max(1))]]
    checks.append(_check("lifted start / end poses feasible with the hard limits (the path of the first PTP is "
                         "not collision-checked)", all(end_ok), f"start {end_ok[0]}, end {end_ok[1]}", "both"))
    try:
        fp = first_ptp(ct, robot, cfg)
        checks.append(_check(f"first PTP peak TCP speed (case '{fp.case}', kukasim/first_ptp.py)",
                             fp.peak_tcp_mm_s <= fp.limit_mm_s,
                             f"{fp.peak_tcp_mm_s:.1f} mm/s at $VEL_AXIS {fp.vel_axis_pct} %",
                             f"<= {fp.limit_mm_s:g} mm/s"))
    except StartPoseError as e:
        checks.append(_check("first PTP peak TCP speed (kukasim/first_ptp.py)", False, "n/a", "", str(e)))
    # speeds
    v_s = p["straight_speed_mm_s"]
    v_cmax = v_s * float(ct.r_eff.max())
    vmax_lim = float(p["max_tcp_speed_mm_s"])
    checks.append(_check("TCP speed: straight / max corner / $VEL.CP at 100 % override",
                         max(v_s, v_cmax, ct.v_program) <= vmax_lim,
                         f"{v_s:.1f} / {v_cmax:.1f} / {ct.v_program:.1f} mm/s", f"<= {vmax_lim:g} mm/s"))
    is_c = np.array([k == "corner" for k in ct.kinds])
    if ct.mode == "pc":
        ovr = ct.ov_corner[is_c]
        checks.append(_check("corner override inside 1..100 %", ((ovr >= 1) & (ovr <= 100)).all(),
                             f"{ovr.min():.0f}..{ovr.max():.0f} %", "1..100 %"))
    else:
        vc = ct.v_corner_cmd[is_c]
        checks.append(_check("corner $VEL.CP inside 0.1 mm/s .. max TCP speed", ((vc >= 0.1) & (vc <= vmax_lim)).all(),
                             f"{vc.min():.2f}..{vc.max():.2f} mm/s", f"0.1..{vmax_lim:g} mm/s"))
    used = ct.r_cmd[is_c & ~ct.excluded]
    counts = {r: int(np.isclose(used, r).sum()) for r in p["ratios"]}
    checks.append(_check("schedule: every ratio used at least min_repeats times",
                         min(counts.values()) >= p["min_repeats"],
                         ", ".join(f"{r:g}: {n}" for r, n in counts.items()), f">= {p['min_repeats']} each"))
    qerr = np.abs(ct.r_eff - ct.r_cmd)[is_c]
    checks.append(_check("quantisation error of r (pc: integer $OV_PRO %; krl: $VEL.CP 0.01 mm/s)", qerr.max() <= 0.01,
                         f"max {qerr.max():.3f}", "<= 0.01", severity="warning"))
    ex = np.nonzero(ct.excluded)[0]
    checks.append(_check(f"min straight between corner zones ({p['min_straight_mm']:g} mm)", len(ex) == 0,
                         f"min {np.nanmin(np.r_[ct.straight_in, ct.straight_out]):.1f} mm, "
                         f"{len(ex)} corner(s) set to r = 1",
                         f">= {p['min_straight_mm']:g} mm",
                         "excluded corners: " + ", ".join(str(int(j) - 1) for j in ex) if len(ex) else "",
                         severity="warning"))
    dev = np.abs(ct.ratio_pred - ct.r_eff)[is_c]
    jdev = np.nonzero(is_c)[0][np.argmax(dev)] - 1 if len(dev) else -1
    checks.append(_check("model: predicted corner/straight ratio reaches the commanded ratio",
                         dev.max() <= 0.02, f"max deviation {dev.max():.3f} (corner {jdev})", "<= 0.02",
                         "window too short for the ramp + PC latency, or the blend radius caps the speed",
                         severity="warning"))
    ct.checks = checks
    return checks


# -------------------------------------------------------------------- outputs
def _abc_rows(ct, robot):
    """KRL ABC per waypoint from FK of the planned joints (falls back to the commanded orientation)."""
    res = ct.plan
    rows = []
    for j in range(len(ct.W)):
        k = int(np.argmin(np.abs(ct.dense["s_prog"] - ct.s_w[j])))
        q = res.q[k]
        if np.isnan(q[0]):
            T = tcp_poses(ct.W[j][None], np.array([[0, 0, -1.0]]),
                          np.radians([ct.params["theta_deg"]]))[0, 0]
            rows.append((None, matrix_to_abc(T[:3, :3], c_hint_deg=0.0)))
        else:
            T = robot.fk(q)
            rows.append((np.degrees(q), matrix_to_abc(T[:3, :3], c_hint_deg=0.0)))
    # round away -0.0 and 1e-12 noise so the KRL / CSV values read cleanly
    return [(q, tuple(float(np.round(v, 6)) + 0.0 for v in abc)) for q, abc in rows]


def write_waypoints_csv(path, ct, robot):
    import csv
    rows = _abc_rows(ct, robot)
    hdr = ["wp", "kind", "corner_id", "hilbert_vertex", "x", "y", "z", "A", "B", "C",
           "A1", "A2", "A3", "A4", "A5", "A6", "turn", "s_prog_mm", "seg_in_mm", "seg_out_mm",
           "straight_in_mm", "straight_out_mm", "excluded_min_straight", "ratio_cmd", "override_pct",
           "ratio_quantised", "blend_half_mm", "blend_min_radius_mm", "model_v_straight_mm_s",
           "model_v_corner_mm_s", "model_ratio", "model_only_width_rel"]
    cid = ct.corner_ids
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for j in range(len(ct.W)):
            q, abc = rows[j]
            c = ct.kinds[j] == "corner"
            fmt = lambda v, n=2: "" if v is None or not np.isfinite(v) else f"{v:.{n}f}"
            w.writerow([j, ct.kinds[j], cid[j], int(ct.vertex_index[j]), *[f"{v:.3f}" for v in ct.W[j]],
                        fang(abc[0], 4), fnum(abc[1], 4), fang(abc[2], 4),
                        *([fnum(v, 4) for v in q] if q is not None else [""] * 6),
                        {1: "L", -1: "R"}.get(int(ct.turn[j]), ""), f"{ct.s_w[j]:.2f}",
                        fmt(ct.seg[j - 1]) if j > 0 else "", fmt(ct.seg[j]) if j < len(ct.seg) else "",
                        fmt(ct.straight_in[j]), fmt(ct.straight_out[j]), int(ct.excluded[j]),
                        f"{ct.r_cmd[j]:.3f}" if c else "", f"{ct.ov_corner[j]:.0f}",
                        f"{ct.r_eff[j]:.4f}" if c else "", fmt(ct.d_eff[j], 3) if c else "",
                        fmt(ct.r_blend[j], 3) if c else "", fmt(ct.v_straight[j], 3),
                        fmt(ct.v_corner[j], 3), fmt(ct.ratio_pred[j], 4) if c else "",
                        fmt(1.0 / ct.ratio_pred[j], 3) if c else ""])


def write_schedule_csv(path, ct):
    """One row per corner. pc mode: for a PC-side override controller (not in this repository):
    switch to override_corner_pct when the TCP passes the window start, back to
    override_straight_pct at the window end. krl mode: the zones and corner $VEL.CP that the
    program itself sets (override 100 %); for reference and for reading the logs."""
    import csv
    s_prog, t = ct.dense["s_prog"], ct.dense["t"]
    pc = ct.mode == "pc"
    ov_s = ct.params["straight_override_pct"] if pc else 100
    zb, za = _zones(ct.params)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corner_id", "wp", "x", "y", "turn", "dir_in", "dir_out", "ratio_cmd", "excluded",
                    "speed_mode", "override_straight_pct", "override_corner_pct", "vel_cp_straight_mm_s",
                    "vel_cp_corner_mm_s", "s_vertex_mm", "s_window_start_mm",
                    "s_window_end_mm", "window_start_x", "window_start_y", "window_end_x", "window_end_y",
                    "model_t_window_start_s", "model_t_vertex_s", "model_t_window_end_s", "model_ratio"])
        for j, k in enumerate(ct.kinds):
            if k != "corner":
                continue
            uin = (ct.W[j] - ct.W[j - 1]) / ct.seg[j - 1]
            uout = (ct.W[j + 1] - ct.W[j]) / ct.seg[j]
            a = ct.W[j] - zb * uin
            b = ct.W[j] + za * uout
            ts = [float(np.interp(x, s_prog, t)) for x in (ct.windows[j, 0], ct.s_w[j], ct.windows[j, 1])]
            dname = lambda u: DIRS4.get(tuple(np.round(u[:2]).astype(int)), "")
            w.writerow([j - 1, j, f"{ct.W[j, 0]:.3f}", f"{ct.W[j, 1]:.3f}", {1: "L", -1: "R"}[int(ct.turn[j])],
                        dname(uin), dname(uout), f"{ct.r_cmd[j]:.3f}", int(ct.excluded[j]), ct.mode, f"{ov_s:.0f}",
                        f"{ct.ov_corner[j] if pc else 100:.0f}", f"{ct.v_program:.2f}",
                        f"{ct.v_program if pc else ct.v_corner_cmd[j]:.2f}",
                        f"{ct.s_w[j]:.2f}", f"{ct.windows[j, 0]:.2f}",
                        f"{ct.windows[j, 1]:.2f}", f"{a[0]:.3f}", f"{a[1]:.3f}", f"{b[0]:.3f}", f"{b[1]:.3f}",
                        *[f"{v:.3f}" for v in ts], f"{ct.ratio_pred[j]:.4f}"])


def write_speed_csv(path, ct, step=5):
    import csv
    d = ct.dense
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["s_prog_mm", "s_path_mm", "x", "y", "model_t_s", "model_v_cmd_mm_s", "model_v_mm_s"])
        keep = np.unique(np.r_[np.arange(0, len(d["v"]), step), np.nonzero(d["vert"] >= 0)[0], len(d["v"]) - 1])
        for i in keep:
            w.writerow([f"{d['s_prog'][i]:.2f}", f"{d['s_path'][i]:.2f}", f"{d['P'][i, 0]:.3f}",
                        f"{d['P'][i, 1]:.3f}", f"{d['t'][i]:.3f}", f"{d['v_cmd'][i]:.3f}", f"{d['v'][i]:.3f}"])


CHECKED_SCOPE = "checked against cell.yaml only; site keep-outs must be checked locally before running"
CONTROLLER_DEFAULTS = ("; Assumes the controller's INI / BAS(#INITMOV,0) defaults for everything not set here "
                       "($ACC.CP, $VEL.ORI, $ACC.ORI, ...); $APO.CDIS is set explicitly.")


def _vel(v_mm_s):
    return f"  $VEL.CP = {v_mm_s / 1000.0:.5f}"


def first_ptp(ct, robot, cfg):
    """$VEL_AXIS of the approach PTP (the first PTP, from the pose the robot is in): see
    kukasim/first_ptp.py. Never above approach_vel_axis_pct."""
    p = ct.params
    return plan_first_ptp(robot, cfg, ct.approach_q, p["approach_acc_axis_pct"],
                          pct_max=int(p["approach_vel_axis_pct"]))


def write_krl(path, ct, robot, cfg, name="hilbert_corner"):
    """KRL export: one LIN per waypoint (no densification, so C_DIS acts at the corners);
    in krl speed mode every corner leg is split at +-d_zone and the corner sub-segments get
    $VEL.CP = r x v. Written with LF line endings on every platform."""
    p = ct.params
    rows = _abc_rows(ct, robot)
    tool = cfg.tool_xyzabc
    lift = float(p["approach_lift_mm"])
    z0, kx, ky, x0, y0 = ct.z_law
    zr = (ct.W[:, 2].min(), ct.W[:, 2].max())
    zdesc = f"layer Z {z0:.3f}" if kx == 0 and ky == 0 else \
        f"layer Z = {z0:.3f} + {kx:g} (X - {x0:.1f}) + {ky:g} (Y - {y0:.1f}) ({zr[0]:.3f}..{zr[1]:.3f})"
    v_s = float(p["straight_speed_mm_s"])
    fp = first_ptp(ct, robot, cfg)
    lines = [
        NOT_RUN,
        "; This file was produced by kuka-sim (Hilbert corner test) and has never been run.",
        "; Do not load it on a controller without a separate review and a dry run.",
        f"; Hilbert order {p['order']}, side {ct.side:g} mm, centre ({ct.center[0]:.1f}, {ct.center[1]:.1f}), "
        f"{zdesc}, {sum(k == 'corner' for k in ct.kinds)} corners",
        f"; TOOL 3 assumed = {{X {tool[0]}, Y {tool[1]}, Z {tool[2]}, A {tool[3]}, B {tool[4]}, C {tool[5]}}}"
        " (copied, to be re-confirmed); BASE 0",
    ]
    if ct.mode == "krl":
        lines += [
            f"; Speed (krl mode): $VEL.CP = {v_s:g} mm/s on the straights; each corner leg is split at "
            f"+-{p['d_zone_mm']:g} mm",
            ";   around the vertex and the two corner sub-segments run at $VEL.CP = r x v (set in this program,",
            ";   path-synchronous). Speeds assume $OV_PRO 100 %; the ratios hold at any constant override.",
        ]
    else:
        lines += [
            f"; Speed (pc mode): $VEL.CP = {ct.v_program:.1f} mm/s at 100 % override; straights at $OV_PRO "
            f"{p['straight_override_pct']:g} % = {v_s:g} mm/s.",
            "; The corner overrides are NOT set here: a PC-side schedule follower (NOT provided in this",
            ";   repository) must switch $OV_PRO per schedule.csv.",
        ]
    lines += [
        f"; C_DIS {p['c_dis_mm']:g} mm; KRC limits it to half the shorter adjacent segment.",
        CONTROLLER_DEFAULTS,
        f"; Approach: PTP to {lift:g} mm above the start at $VEL_AXIS {fp.vel_axis_pct:d} % / "
        f"$ACC_AXIS {p['approach_acc_axis_pct']:g} % (configured at most {p['approach_vel_axis_pct']:g} %); "
        "its path is NOT collision-checked.",
        *fp.header(),
        f"; Checks: {'passed' if ct.ok else 'FAILED - see report.md'} ({CHECKED_SCOPE})",
    ]
    if not ct.ok:
        lines.append("; CHECKS FAILED: this program is outside the simulated safe envelope.")
    lines += [
        f"DEF {krl_name(name)}()",
        "  $BASE = $NULLFRAME",
        "  $TOOL = TOOL_DATA[3]",
        _vel(ct.v_program),
        f"  $APO.CDIS = {p['c_dis_mm']:g}",
        "  $ORI_TYPE = #CONSTANT",
    ]
    lines += [f"  $VEL_AXIS[{i}] = {fp.vel_axis_pct:d}" for i in range(1, 7)]   # first PTP (first_ptp.py)
    lines += [f"  $ACC_AXIS[{i}] = {p['approach_acc_axis_pct']:g}" for i in range(1, 7)]

    def lin(pos, abc):
        return "  LIN " + lin_pose(pos, abc)

    if ct.approach_q is not None:
        lines.append("  PTP " + ptp_axes(np.degrees(ct.approach_q))
                     + f"  ; {lift:g} mm above the start; first PTP: "
                     + ("speed bounded from any start pose" if fp.case == "bound"
                        else "from the DECLARED START pose (see header)"))
    else:
        lines.append("  ; no IK solution within the hard limits for the lifted start pose: approach PTP omitted")
        lines.append(lin((ct.W[0] + [0, 0, lift]), rows[0][1]))
    cid = ct.corner_ids
    for j in range(len(ct.W)):
        abc = rows[j][1]
        if ct.kinds[j] != "corner":
            lines.append(lin(ct.W[j], abc) + f"  ; {ct.kinds[j]} (exact stop)")
            continue
        note = f"  ; corner {cid[j]}: r {ct.r_cmd[j]:g}"
        if ct.excluded[j]:
            note += " (too close to a neighbour, r = 1)"
        if ct.mode == "pc":
            lines.append(lin(ct.W[j], abc) + " C_DIS" + note + f", OV {ct.ov_corner[j]:.0f} %")
            continue
        uin = (ct.W[j] - ct.W[j - 1]) / ct.seg[j - 1]
        uout = (ct.W[j + 1] - ct.W[j]) / ct.seg[j]
        d = float(p["d_zone_mm"])
        lines.append(lin((ct.W[j] - d * uin), abc) + f" C_DIS  ; zone start corner {cid[j]}")
        lines.append(_vel(ct.v_corner_cmd[j]))
        lines.append(lin(ct.W[j], abc) + " C_DIS" + note + f", v {ct.v_corner_cmd[j]:.2f} mm/s")
        lines.append(lin((ct.W[j] + d * uout), abc) + f" C_DIS  ; zone end corner {cid[j]}")
        lines.append(_vel(v_s))
    lines.append(lin((ct.W[-1] + [0, 0, lift]), rows[-1][1]))
    lines.append("END")
    with open(path, "w", newline="\n") as fh:          # LF on every platform (K-2)
        fh.write("\n".join(lines) + "\n")


def ratio_colors(r):
    """Diverging: r < 1 orange (wider bead), r > 1 blue (thinner), r = 1 gray."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("ratio", ["#c4561f", "#f1b48f", "#bdbcb6", "#9cc0ec", "#1f5fae"])
    lo = min(0.99, float(np.min(r)))
    hi = max(1.01, float(np.max(r)))
    return cmap, TwoSlopeNorm(1.0, lo, hi)


def plot_path(path, ct, cfg, coll=None):
    from . import outputs  # noqa: F401  (matplotlib backend and style)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    bed = cfg.bed()
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    if ct.reach is not None:
        xs, ys, feas = ct.reach
        res = xs[1] - xs[0]
        ax.imshow(np.where(feas, np.nan, 1.0), origin="lower", cmap="Greys", vmin=0, vmax=4,
                  extent=[xs[0] - res / 2, xs[-1] + res / 2, ys[0] - res / 2, ys[-1] + res / 2],
                  interpolation="nearest", zorder=0)
    ax.add_patch(Rectangle((bed["x"][0], bed["y"][0]), bed["x"][1] - bed["x"][0], bed["y"][1] - bed["y"][0],
                           fill=False, ec="#8a8984", lw=1.0))
    if coll is not None:
        for b in coll.boxes[1:]:
            ax.add_patch(Rectangle(b.lo[:2], *(b.hi[:2] - b.lo[:2]), fill=False, ec="#8a8984", lw=0.8, ls="--"))
            ax.text(*(0.5 * (b.lo[:2] + b.hi[:2])), b.name.replace("_", " "), fontsize=5.5, color="#52514e",
                    ha="center", va="center", rotation=90 if b.hi[1] - b.lo[1] > 300 else 0)
    P = ct.dense["P"]
    ax.plot(P[:, 0], P[:, 1], color="#0b0b0b", lw=0.9, zorder=2)
    ax.plot(*ct.W[0, :2], "o", color="#0b0b0b", ms=5, zorder=3)
    ax.plot(*ct.W[-1, :2], "s", color="#0b0b0b", ms=5, zorder=3)
    ax.annotate("start", ct.W[0, :2], xytext=(-6, -10), textcoords="offset points", fontsize=7, ha="right")
    ax.annotate("end", ct.W[-1, :2], xytext=(6, -10), textcoords="offset points", fontsize=7)
    is_c = np.array([k == "corner" for k in ct.kinds])
    cmap, norm = ratio_colors(ct.r_cmd[is_c])
    sc = ax.scatter(ct.W[is_c, 0], ct.W[is_c, 1], c=ct.r_cmd[is_c], cmap=cmap, norm=norm, s=70,
                    edgecolors="#fcfcfb", linewidths=1.5, zorder=4)
    step = ct.side / ((1 << int(ct.params["order"])) - 1)
    off = 0.16 * step
    for j in np.nonzero(is_c)[0]:
        uin = (ct.W[j] - ct.W[j - 1])[:2] / ct.seg[j - 1]
        uout = (ct.W[j + 1] - ct.W[j])[:2] / ct.seg[j]
        o = (uin - uout) * off          # outside of the corner
        lab = f"{ct.corner_ids[j]}\n{ct.r_cmd[j]:g}" + ("*" if ct.excluded[j] else "")
        ax.text(ct.W[j, 0] + o[0], ct.W[j, 1] + o[1], lab, fontsize=6, ha="center", va="center",
                color="#0b0b0b", linespacing=0.9, zorder=5)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.8)
    cb.set_label("corner override ratio r")
    ax.set_aspect("equal")
    ax.set_xlabel("X [mm] (BASE 0)")
    ax.set_ylabel("Y [mm]")
    pad = 15
    ax.set_xlim(bed["x"][0] - pad - 20, bed["x"][1] + pad)
    ax.set_ylim(bed["y"][0] - pad, bed["y"][1] + pad)
    n_c = int(is_c.sum())
    ax.set_title(f"Hilbert corner test: order {ct.params['order']}, side {ct.side:g} mm, {n_c} corners. "
                 "Labels: corner id / r (* = too close, r = 1)\n"
                 "Solid: bed outline; dashed: clamp boxes (inflated); gray: not feasible for the vertical nozzle",
                 loc="left", fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_speed(path, ct):
    from . import outputs  # noqa: F401
    import matplotlib.pyplot as plt
    d = ct.dense
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(d["s_prog"], d["v_cmd"], color="#bdbcb6", lw=1.0, label="commanded ($VEL.CP x override)")
    ax.plot(d["s_prog"], d["v"], color="#2a78d6", lw=1.5, label="model estimate")
    ax.set_xlabel("programmed path length [mm]")
    ax.set_ylabel("TCP speed [mm/s]")
    ax.set_ylim(0, None)
    ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=7)
    ax.set_title("Corner test: TCP speed along the path (model estimate; compare with the logged $VEL_ACT)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cdis_sensitivity(ct, cdis_values=(0.5, 1.0, 2.0, 3.0, 5.0)):
    """Highest corner speed ratio the blend radius allows (90 deg corner, model):
    v <= sqrt(a_lat * d / sqrt(2)) -> r_max = that / v_straight."""
    a = float(ct.params["lateral_acc_mm_s2"])
    v_s = ct.params["straight_speed_mm_s"]
    return [(c, float(np.sqrt(a * c / np.sqrt(2)) / v_s)) for c in cdis_values]


def write_report(path, ct, cfg, files):
    p = ct.params
    is_c = np.array([k == "corner" for k in ct.kinds])
    n_c = int(is_c.sum())
    step = ct.side / ((1 << int(p["order"])) - 1)
    L = []
    L.append("# Hilbert corner test (simulation only)\n")
    L.append("Generated by `python scripts/run_hilbert_corner.py`. Nothing here was run on the robot; the KRL "
             "file carries the NOT RUN header. Every speed below is a **model estimate** unless it is a "
             "commanded value.\n")
    L.append(f"**Scope of the checks:** {CHECKED_SCOPE}. The obstacles are the clamp estimates and the optional "
             "`keep_out` list in `config/cell.yaml` (empty in the repository).\n")
    L.append("## Layout\n")
    L.append(f"- Hilbert curve order {p['order']}: {len(ct.points)} vertices, grid step {step:.2f} mm, "
             f"side {ct.side:g} mm, centre ({ct.center[0]:.1f}, {ct.center[1]:.1f}) mm in BASE 0.")
    cap = f", capped at {p['max_side_mm']:g} mm" if p.get("max_side_mm") is not None else ""
    fitted = "fitted: largest square (rounded down to 10 mm) whose points are all reachable and collision-free " \
             f"with the vertical nozzle, at least {p['fit_margin_mm']:g} mm from any infeasible point and the " \
             f"bed edge and at least {p['clamp_margin_mm']:g} mm (XY) from the inflated clamp boxes " \
             f"(map grid {p['fit_grid_mm']:g} mm){cap}" if ct.reach is not None else "set by hand"
    L.append(f"- Side and centre: {fitted}. The unreachable +X/-Y bed corner and the clamp boxes are outside it.")
    z0, kx, ky, x0, y0 = ct.z_law
    zsrc = f"bed top_z {cfg.bed()['top_z']} + layer height {cfg.layer_height}" if p["layer_z"]["z0"] is None \
        else "set in layer_z"
    L.append(f"- Layer Z: z(x, y) = z0 + kx (x - x0) + ky (y - y0) with z0 = {z0:.2f} mm ({zsrc}), kx = {kx:g}, "
             f"ky = {ky:g}, (x0, y0) = ({x0:.1f}, {y0:.1f}); Z {ct.W[:, 2].min():.2f} .. {ct.W[:, 2].max():.2f} mm "
             f"over the path. Nozzle vertical, rotation {p['theta_deg']:g} deg about it (A0 B90 C0 for 0), TOOL 3.")
    L.append(f"- Path length {ct.s_w[-1]:.0f} mm, {n_c} corners ({int((ct.turn == 1).sum())} left, "
             f"{int((ct.turn == -1).sum())} right turns); model print time {ct.dense['t'][-1]:.0f} s.\n")
    L.append("## Corner speed schedule\n")
    if ct.mode == "krl":
        L.append(f"- Mode `krl` (default): set in the program itself, path-synchronous. Straight: $VEL.CP = "
                 f"{p['straight_speed_mm_s']:g} mm/s. Every corner leg is split at +-{p['d_zone_mm']:g} mm around "
                 f"the vertex (d_zone >= 2 x C_DIS = {2 * p['c_dis_mm']:g} mm) and the two corner sub-segments run at "
                 "$VEL.CP = r x v, back to v after them. Speeds assume $OV_PRO 100 %; the ratios hold at any constant "
                 "override. Extruder rpm fixed. `schedule.csv` lists the zones for reading the logs.")
        L.append("- Mode `pc` (option, `speed_mode: pc`): one $VEL.CP and a PC-side controller switching $OV_PRO per "
                 "`schedule.csv`. That needs a PC-side schedule follower, which this repository does not provide.")
    else:
        L.append(f"- Mode `pc`: straight {p['straight_speed_mm_s']:g} mm/s at $OV_PRO {p['straight_override_pct']:g} % "
                 f"($VEL.CP = {ct.v_program:.1f} mm/s at 100 %). Extruder rpm fixed.")
        L.append(f"- Corner: $OV_PRO = round(r x {p['straight_override_pct']:g}) %, from {p['window_before_mm']:g} mm "
                 f"before to {p['window_after_mm']:g} mm after the vertex (`schedule.csv`). This needs a PC-side "
                 "schedule follower, which this repository does NOT provide.")
    L.append(f"- Ratios {', '.join(f'{r:g}' for r in p['ratios'])}, each at least {p['min_repeats']}x, "
             f"seeded random order (seed {p['seed']}).")
    L.append(f"- Minimum straight between corner zones: {p['min_straight_mm']:g} mm. **Assumption:** the "
             "extruder flow responds within a few tenths of a second; at about 12 mm/s 0.3 s is about 4 mm, so "
             "15 mm (about 4 response times) keeps one corner's transient out of the next. Shorter -> r = 1.")
    s_all = np.r_[ct.straight_in[is_c], ct.straight_out[is_c]]
    L.append(f"- Straight lengths between corner zones: {np.nanmin(s_all):.1f} .. {np.nanmax(s_all):.1f} mm; "
             f"vertex-to-vertex segments {ct.seg.min():.1f} .. {ct.seg.max():.1f} mm.")
    ex = [int(j) - 1 for j in np.nonzero(ct.excluded)[0]]
    L.append(f"- Corners set to r = 1 because a neighbour is too close: {', '.join(map(str, ex)) if ex else 'none'}.\n")
    L.append("| r | corners |\n|---|---|")
    for r in p["ratios"]:
        ids = [str(ct.corner_ids[j]) for j in np.nonzero(is_c & ~ct.excluded & np.isclose(ct.r_cmd, r))[0]]
        L.append(f"| {r:g} | {', '.join(ids)} ({len(ids)}) |")
    L.append("")
    L.append("## Checks over the whole path\n")
    L.append("| check | result | value | limit |\n|---|---|---|---|")
    for c in ct.checks:
        res = "ok" if c["ok"] else ("**FAIL**" if c["severity"] == "error" else "warning")
        val = c["value"] + (f" ({c['detail']})" if c["detail"] and not c["ok"] else "")
        L.append(f"| {c['name']} | {res} | {val} | {c['limit']} |")
    L.append("")
    L.append("## Corner-speed model (estimate)\n")
    lat = (f"each override switch takes effect {p['pc_latency_s']:g} s late (assumed)" if ct.mode == "pc"
           else "the corner $VEL.CP applies from the zone start (no PC latency)")
    L.append("Assumptions: the C_DIS blend is the quadratic Bezier through the vertex as control point, of half-"
             f"length d = min(C_DIS {p['c_dis_mm']:g} mm, half the shorter adjacent segment); speed follows "
             f"the commanded speed with a trapezoid profile at {p['path_acc_mm_s2']:g} mm/s^2 (assumed), is capped "
             f"by sqrt(a_lat R) with a_lat {p['lateral_acc_mm_s2']:g} mm/s^2 (assumed); {lat}; inside each blend "
             f"the commanded speed is scaled by the corner dip factor {p['corner_dip_factor']:g} (1.0 = no dip). "
             "None of these numbers was read from the controller.\n")
    rp = ct.ratio_pred[is_c]
    d = np.abs(rp - ct.r_eff[is_c])
    L.append(f"- Blend half-length {ct.d_eff[is_c].min():g} mm, min radius {np.min(ct.r_blend[is_c]):.2f} mm; "
             f"cap sqrt(a_lat R) = {np.sqrt(p['lateral_acc_mm_s2'] * np.min(ct.r_blend[is_c])):.1f} mm/s.")
    L.append(f"- Predicted corner/straight ratio: {np.nanmin(rp):.3f} .. {np.nanmax(rp):.3f}; max deviation from "
             f"the commanded (quantised) ratio {d.max():.3f}. Per corner: `waypoints.csv` (`model_ratio`), "
             "speed profile: `speed_profile.csv`, `speed.png`.")
    L.append(("- With the dip factor at 1.0 nothing limits the corner speed at the default settings, so the model "
              "simply returns the commanded ratio. " if p["corner_dip_factor"] == 1.0 else
              f"- The dip factor {p['corner_dip_factor']:g} is an assumption, not a measurement. ") +
             "Real controllers do slow down below the programmed speed in "
             "blended 90 deg corners; the model does not know by how much (`corner_dip_factor` is a placeholder), "
             "nor the KRC's own ramps or look-ahead.")
    L.append("- **The logged speed decides.** The corner speed and ratio actually reached come from the controller "
             "log ($VEL_ACT or a trace), not from this model.")
    L.append("- Predicted corner bead width relative to the straight bead (**model only**: flow / speed at fixed rpm, "
             "no flow dynamics, no corner dip unless set) = 1 / model ratio: "
             f"{1 / np.nanmax(rp):.2f} .. {1 / np.nanmin(rp):.2f} (`model_only_width_rel` in `waypoints.csv`).")
    L.append("- Sensitivity to C_DIS (model): highest corner ratio before the blend radius caps the speed: " +
             ", ".join(f"C_DIS {c:g} mm -> {r:.2f}" for c, r in cdis_sensitivity(ct)) + ".\n")
    L.append("## What the lab still has to measure\n")
    L.append("- The real TCP speed through each corner ($VEL_ACT or the trace at the controller's IPO rate) "
             "against `model_ratio`; the real ramp / blend behaviour replaces the assumed accelerations.")
    if ct.mode == "pc":
        L.append("- The real delay from the PC's override command to the speed change (`pc_latency_s`).")
    L.append("- The extruder flow response time behind the 15 mm minimum straight.")
    L.append("- Corner bead width per corner id (and straight bead width next to it) to get width vs r.")
    L.append("- TOOL 3 re-confirmation, the bed top over the square (top_z varies 0.95 .. 1.38 mm) and the "
             "clamp positions (photo estimates) before any run.")
    L.append("- Site keep-outs (hopper, hose, cables, cage, anything not in cell.yaml): added to a local copy of "
             "`config/cell.yaml` (`keep_out`) and checked locally before running; never committed.")
    L.append("- The controller's INI / BAS(#INITMOV,0) defaults the program relies on ($ACC.CP, $VEL.ORI, ...).\n")
    L.append("## Files\n")
    for f, what in files:
        L.append(f"- `{f}`: {what}")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")

"""Dynamic programming over (toolpath point x redundancy-angle bin x IK configuration).

State at point i: (angle bin b, IK slot k) with joint vector q. The TCP pose is
R = R_ref(d_i) @ Rx(theta_b): tool +X along the nozzle direction d_i, theta the
rotation about it. theta = 0 puts tool +Z (the A6 axis, out of the flange) along
the projection of BASE +X, which for a vertical nozzle is the planar programs'
A0 B90 C0.

Hard constraints (a state or transition that violates one does not exist):
  * joint limits shrunk by the margin from cell.yaml,
  * optional |A5| >= min_abs_a5,
  * clearance >= 0 against the bed (with gap), the clamp boxes and the floor,
  * |dq_j| <= velocity_scale * vmax_j * ds / v_print between neighbours,
  * |delta bin| <= max_dbin.
Costs: node = w_l * limit + w_s * singularity; edge = w_v * sum((dq/dq_max)^2).
Where no continuous chain exists the path is split, and the split is reported.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .collision import CollisionModel
from .rotations import make_transform, rx


def reference_frames(dirs):
    """(N,3) nozzle directions -> (N,3,3) frames with x = d, z = BASE +X projected."""
    d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    ex = np.array([1.0, 0, 0])
    z = ex - (d @ ex)[:, None] * d
    zn = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(zn < 1e-6):
        raise ValueError("nozzle direction parallel to BASE X; reference undefined")
    z = z / zn
    y = np.cross(z, d)
    return np.stack([d, y, z], axis=-1)


def tcp_poses(points, dirs, thetas):
    """(N,3),(N,3),(B,) -> (N,B,4,4)."""
    Rref = reference_frames(dirs)
    R = Rref[:, None] @ rx(thetas)[None]
    return make_transform(R, np.broadcast_to(points[:, None, :], R.shape[:-2] + (3,)))


@dataclass
class States:
    """Feasible states at one toolpath point."""
    bins: np.ndarray       # (n,)
    slots: np.ndarray      # (n,)
    q: np.ndarray          # (n, 6) rad
    node: np.ndarray       # (n,)
    limit_margin: np.ndarray   # (n,) deg, min over joints to the margin-reduced limit
    clearance: np.ndarray  # (n,) mm


@dataclass
class PlanResult:
    toolpath: object
    thetas_deg: np.ndarray
    q: np.ndarray                 # (N, 6) rad, NaN where infeasible
    bins: np.ndarray              # (N,) chosen bin, -1 where infeasible
    slots: np.ndarray
    cost: float
    node_cost: np.ndarray
    limit_margin: np.ndarray      # deg
    clearance: np.ndarray         # mm
    feas_count: np.ndarray        # (N, B) number of feasible IK configs per bin
    segments: list                # [(i0, i1)] continuous planned stretches (inclusive)
    no_state: list                # [(i0, i1)] stretches with no feasible state at all
    breaks: list                  # [i]: states exist at i but none reachable from i-1
    dq_max: np.ndarray            # (N, 6) rad, allowed step into point i
    runtime_s: dict = field(default_factory=dict)
    n_states: np.ndarray = None

    @property
    def feasible(self):
        return ~np.isnan(self.q[:, 0])

    @property
    def complete(self):
        return len(self.segments) == 1 and not self.no_state and \
            self.segments[0] == (0, len(self.bins) - 1)


class Planner:
    def __init__(self, robot, cfg, collision: CollisionModel | None = None):
        self.robot = robot
        self.cfg = cfg
        self.p = cfg.planner["planner"]
        self.coll = collision or CollisionModel(robot, cfg)
        m = np.radians(cfg.limit_margin_deg)
        self.hard_limits = robot.limits + np.array([m, -m])
        nb = int(round(360 / self.p["angle_bin_deg"]))
        self.thetas = np.radians(-180 + np.arange(nb) * self.p["angle_bin_deg"])

    # ----------------------------------------------------------- costs
    def node_cost(self, q):
        lim = self.hard_limits
        md = np.degrees(np.minimum(q - lim[:, 0], lim[:, 1] - q))     # (...,6)
        zl = self.p["limit_soft_zone_deg"]
        c_lim = (np.maximum(0, 1 - md / zl) ** 2).sum(-1)
        za = self.p["a5_soft_zone_deg"]
        c_sing = np.maximum(0, 1 - np.abs(np.degrees(q[..., 4])) / za) ** 2
        w = self.p["weights"]
        return w["limit"] * c_lim + w["singularity"] * c_sing, md.min(-1)

    def edge_cost(self, dq, dq_max):
        return self.p["weights"]["smooth"] * ((dq / dq_max) ** 2).sum(-1)

    def step_limits(self, points):
        ds = np.r_[np.nan, np.linalg.norm(np.diff(points, axis=0), axis=1)]
        dt = ds / self.cfg.print_speed
        return self.p["velocity_scale"] * self.robot.vmax[None] * dt[:, None]

    def bin_dist(self, b1, b2):
        nb = len(self.thetas)
        d = np.abs(b1 - b2) % nb
        return np.minimum(d, nb - d)

    # ---------------------------------------------------------- states
    def build_states(self, toolpath, chunk=60):
        pts, dirs = toolpath.points, toolpath.dirs
        N, B = len(pts), len(self.thetas)
        states = []
        feas = np.zeros((N, B), int)
        min_a5 = np.radians(self.p["min_abs_a5_deg"])
        for i0 in range(0, N, chunk):
            i1 = min(N, i0 + chunk)
            T = tcp_poses(pts[i0:i1], dirs[i0:i1], self.thetas)        # (n,B,4,4)
            Q, ok = self.robot.ik(T.reshape(-1, 4, 4), limits=self.hard_limits)
            n = i1 - i0
            Q = Q.reshape(n, B, -1, 6)
            ok = ok.reshape(n, B, -1)
            ok &= np.abs(Q[..., 4]) >= min_a5
            idx = np.nonzero(ok)
            qv = Q[idx]
            clr = np.full(len(qv), np.inf)
            if len(qv):
                clr = self.coll.clearance(qv)
            good = clr >= 0
            ii, bb, kk = (a[good] for a in idx)
            qv, clr = qv[good], clr[good]
            node, md = self.node_cost(qv)
            np.add.at(feas, (ii + i0, bb), 1)
            for j in range(n):
                sel = ii == j
                states.append(States(bb[sel], kk[sel], qv[sel], node[sel], md[sel],
                                     np.minimum(clr[sel], 50.0)))
        return states, feas

    # -------------------------------------------------------------- DP
    def dp(self, states, dq_max):
        """Forward DP with traceback; splits where no continuous chain exists."""
        N = len(states)
        max_dbin = self.p["max_dbin"]
        acc = [None] * N
        parent = [None] * N
        segments = []          # [start, end] of continuous stretches
        breaks, no_state = [], []
        prev_ok = False
        for i in range(N):
            S = states[i]
            if len(S.node) == 0:
                if no_state and no_state[-1][1] == i - 1:
                    no_state[-1] = (no_state[-1][0], i)
                else:
                    no_state.append((i, i))
                prev_ok = False
                continue
            if prev_ok:
                P = states[i - 1]
                pi, ci = np.nonzero(self.bin_dist(P.bins[:, None], S.bins[None, :]) <= max_dbin)
                dq = S.q[ci] - P.q[pi]
                keep = (np.abs(dq) <= dq_max[i]).all(-1) & np.isfinite(acc[i - 1][pi])
                pi, ci, dq = pi[keep], ci[keep], dq[keep]
                tot = acc[i - 1][pi] + self.edge_cost(dq, dq_max[i])
                best = np.full(len(S.node), np.inf)
                par = np.full(len(S.node), -1)
                if len(tot):
                    order = np.lexsort((tot, ci))
                    first = np.r_[True, ci[order][1:] != ci[order][:-1]]
                    sel = order[first]
                    best[ci[sel]] = tot[sel]
                    par[ci[sel]] = pi[sel]
                if np.isfinite(best).any():
                    acc[i] = best + S.node
                    parent[i] = par
                    segments[-1][1] = i
                    continue
                breaks.append(i)
            acc[i] = S.node.copy()
            parent[i] = np.full(len(S.node), -1)
            segments.append([i, i])
            prev_ok = True
        choice = np.full(N, -1)
        total = 0.0
        for s0, e in segments:
            k = int(np.argmin(acc[e]))
            total += float(acc[e][k])
            for i in range(e, s0 - 1, -1):
                choice[i] = k
                k = parent[i][k]
        segments = [tuple(sg) for sg in segments]
        return choice, segments, breaks, no_state, total

    # ------------------------------------------------------------ plan
    def plan(self, toolpath):
        t0 = time.perf_counter()
        states, feas = self.build_states(toolpath)
        t1 = time.perf_counter()
        dq_max = self.step_limits(toolpath.points)
        choice, segments, breaks, no_state, total = self.dp(states, dq_max)
        t2 = time.perf_counter()
        N = len(states)
        q = np.full((N, 6), np.nan)
        bins = np.full(N, -1)
        slots = np.full(N, -1)
        node = np.full(N, np.nan)
        lm = np.full(N, np.nan)
        clr = np.full(N, np.nan)
        for i, k in enumerate(choice):
            if k >= 0:
                S = states[i]
                q[i], bins[i], slots[i] = S.q[k], S.bins[k], S.slots[k]
                node[i], lm[i], clr[i] = S.node[k], S.limit_margin[k], S.clearance[k]
        return PlanResult(toolpath, np.degrees(self.thetas), q, bins, slots, total, node, lm, clr,
                          feas, segments, no_state, breaks, dq_max,
                          {"states_s": t1 - t0, "dp_s": t2 - t1, "total_s": t2 - t0},
                          np.array([len(s.node) for s in states]))

    # ------------------------------------------------ brute force (tests)
    def brute_force(self, states, dq_max):
        """Exhaustive minimum over all continuous state sequences (small cases only)."""
        best, best_seq = np.inf, None
        max_dbin = self.p["max_dbin"]
        for seq in itertools.product(*[range(len(s.node)) for s in states]):
            c = states[0].node[seq[0]]
            ok = True
            for i in range(1, len(states)):
                P, S = states[i - 1], states[i]
                dq = S.q[seq[i]] - P.q[seq[i - 1]]
                if np.any(np.abs(dq) > dq_max[i]) or \
                        self.bin_dist(P.bins[seq[i - 1]], S.bins[seq[i]]) > max_dbin:
                    ok = False
                    break
                c += self.edge_cost(dq, dq_max[i]) + S.node[seq[i]]
            if ok and c < best:
                best, best_seq = c, seq
        return best, best_seq


def diagnose_point(planner, point, direction):
    """Why a point has no feasible state: counts after each hard constraint, and
    the obstacle that blocks most of the remaining configurations."""
    T = tcp_poses(point[None], direction[None], planner.thetas).reshape(-1, 4, 4)
    _, ok_raw = planner.robot.ik(T, limits=np.array([[-1e9, 1e9]] * 6))
    _, ok_lim0 = planner.robot.ik(T)
    Q, ok = planner.robot.ik(T, limits=planner.hard_limits)
    ok &= np.abs(Q[..., 4]) >= np.radians(planner.p["min_abs_a5_deg"])
    out = {"reachable": int(ok_raw.sum()), "in_limits": int(ok_lim0.sum()),
           "in_limits_with_margin": int(ok.sum()), "collision_free": 0, "blocking": None}
    if ok.any():
        c, det = planner.coll.clearance(Q[ok], return_detail=True)
        out["collision_free"] = int((c >= 0).sum())
        if out["collision_free"] == 0:
            counts = {k: int((v < 0).sum()) for k, v in det.items()}
            out["blocking"] = max(counts, key=counts.get)
    return out


def diagnose(planner, toolpath, res):
    """One reason per infeasible stretch."""
    out = []
    for a, b in res.no_state:
        d = diagnose_point(planner, toolpath.points[(a + b) // 2], toolpath.dirs[(a + b) // 2])
        if d["reachable"] == 0:
            why = "out of reach"
        elif d["in_limits"] == 0:
            why = "joint limits"
        elif d["in_limits_with_margin"] == 0:
            why = "joint-limit margin"
        else:
            why = f"collision ({d['blocking']})"
        out.append({"points": (a, b), "s_mm": (float(toolpath.s[a]), float(toolpath.s[b])),
                    "xy_mid": toolpath.points[(a + b) // 2][:2].round(0).tolist(), "reason": why, **d})
    for i in res.breaks:
        out.append({"points": (i, i), "s_mm": (float(toolpath.s[i]),) * 2,
                    "xy_mid": toolpath.points[i][:2].round(0).tolist(),
                    "reason": "no continuous transition (reconfiguration needed)"})
    return out


def turn_window(planner, point, direction, max_step_deg=10.0):
    """Longest continuous rotation about the nozzle axis (deg) at a fixed point:
    walk the angle bins in order, allowing the IK state to change only by
    <= max_step_deg per joint between neighbouring bins (A6 turns included, so
    the result can exceed 360). Returns (window_deg, start_theta_deg)."""
    T = tcp_poses(point[None], direction[None], planner.thetas).reshape(-1, 4, 4)
    Q, ok = planner.robot.ik(T, limits=planner.hard_limits)
    c = np.full(ok.shape, -1.0)
    if ok.any():
        c[ok] = planner.coll.clearance(Q[ok])
    ok &= c >= 0
    B = len(planner.thetas)
    thr = np.radians(max_step_deg)
    L = np.where(ok, 1, 0)             # chain length ending at (bin, slot)
    start = np.where(ok, np.arange(B)[:, None], -1)
    best, best_start = int(L.max()), int(np.argmax(L.max(1)))
    Lprev, Sprev = L[0].copy(), start[0].copy()
    for u in range(1, 3 * B):
        b, bp = u % B, (u - 1) % B
        dq = np.abs(Q[b][None, :, :] - Q[bp][:, None, :]).max(-1)      # (prev, cur)
        link = (dq <= thr) & ok[bp][:, None] & ok[b][None, :] & (Lprev[:, None] > 0)
        cand = np.where(link, Lprev[:, None], 0)
        k = cand.argmax(0)
        Lc = np.where(ok[b], cand.max(0) + 1, 0)
        Sc = np.where(cand.max(0) > 0, Sprev[k], np.where(ok[b], u, -1))
        Lc = np.minimum(Lc, 2 * B + 1)
        if Lc.max() > best:
            best = int(Lc.max())
            best_start = int(Sc[Lc.argmax()])
        Lprev, Sprev = Lc, Sc
    win = max(0, best - 1) * planner.p["angle_bin_deg"]
    return float(win), float(np.degrees(planner.thetas[best_start % B])) if best > 0 else np.nan

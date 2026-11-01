"""Primitive collision model: capsules for the robot and tool, boxes for the bed and clamps.

The meshes are not available, so every link is a capsule (segment + radius)
and the obstacles are axis-aligned boxes. All sizes come from
config/planner.yaml (estimated) and config/cell.yaml (measured / estimated).
Self-collision and the printed part are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray
    required_clearance: float = 0.0
    status: str = "estimated"


def build_obstacles(cfg) -> list[Box]:
    bed = cfg.bed()
    cp = cfg.planner["collision"]
    if bed["rotation_about_z_deg"] not in (None, 0, 0.0):
        raise NotImplementedError("rotated bed outline")
    top = bed["top_z"]
    boxes = [Box("bed", np.array([bed["x"][0], bed["y"][0], top - bed["thickness"]]),
                 np.array([bed["x"][1], bed["y"][1], top]),
                 required_clearance=float(cp["bed_gap_mm"]),
                 status=cfg.cell["bed"]["outline_status"])]
    hx, hy = cp["clamp_half_xy"]
    inf = float(cp["clamp_inflate_mm"])
    for ob in cfg.cell.get("obstacles", []):
        h = ob.get("top_above_bed", cp["clamp_default_height"])
        if "pad_xy" in ob:
            cx, cy = ob["pad_xy"]
            lo = [cx - hx - inf, cy - hy - inf, top]
            hi = [cx + hx + inf, cy + hy + inf, top + h]
        elif "x" in ob:
            # y unknown: a band along the whole -X edge of the bed
            x0, x1 = cp["minus_x_clamp_band"]
            lo = [x0 - inf, bed["y"][0] - inf, top]
            hi = [x1 + inf, bed["y"][1] + inf, top + h]
        else:
            continue
        boxes.append(Box(ob["name"], np.array(lo, float), np.array(hi, float), 0.0,
                         ob.get("status", "estimated")))
    return boxes


class CollisionModel:
    """Capsules attached to robot frames; clearance against boxes and the floor."""

    def __init__(self, robot, cfg):
        self.robot = robot
        self.boxes = build_obstacles(cfg)
        cp = cfg.planner["collision"]
        self.cp = cp
        bed = cfg.bed()
        self.floor_z = bed["top_z"] - bed["thickness"] if cp.get("check_floor", True) else -np.inf
        lr = cp["link_radius"]
        t = cp["tool"]
        tx, ty, tz = cfg.tool_xyzabc[:3]
        # tool primitives in the flange frame (segment endpoints, radius)
        # Segment ends are pulled in by the radius so that the capsule's round cap
        # ends where the part ends (a capsule overhangs its segment by r).
        nz0 = tx - t["nozzle_length"]
        rb, rn = t["barrel_radius"], t["nozzle_radius"]
        nz1 = tx - cp["tip_free_mm"] - rn
        self.tool_caps = [
            ("tool_bracket", np.array([0, 0, 0.0]), np.array([0, ty, tz]), t["bracket_radius"]),
            ("tool_barrel", np.array([-t["barrel_back_mm"] + rb, ty, tz]),
             np.array([nz0 - rb, ty, tz]), rb),
            ("tool_nozzle", np.array([nz0, ty, tz]), np.array([max(nz1, nz0), ty, tz]), rn),
            # The tip itself: exempt from the bed/floor (it prints there) but not from clamps.
            ("tool_tip", np.array([max(nz1, nz0), ty, tz]), np.array([tx, ty, tz]), cp.get("tip_radius", 3.0)),
        ]
        self.radii = np.array([lr["link1"], lr["upper_arm"], lr["forearm"], lr["wrist"]]
                              + [c[3] for c in self.tool_caps], float)
        self.names = ["link1", "upper_arm", "forearm", "wrist"] + [c[0] for c in self.tool_caps]

    def capsules(self, q):
        """q (N,6) rad -> segments (N, M, 2, 3) mm."""
        F = self.robot.link_frames(q)
        p1 = F[0][..., :3, 3]
        p2 = F[1][..., :3, 3]
        p3 = F[2][..., :3, 3]
        Tf = F[5]
        wc = self.robot.wrist_center(Tf)
        pf = Tf[..., :3, 3]
        segs = [(p1, p2), (p2, p3), (p3, wc), (wc, pf)]
        for _, a, b, _ in self.tool_caps:
            pa = (Tf[..., :3, :3] @ a) + pf
            pb = (Tf[..., :3, :3] @ b) + pf
            segs.append((pa, pb))
        return np.stack([np.stack(s, -2) for s in segs], -3)

    @staticmethod
    def segment_box_distance(seg, lo, hi, iters=24):
        """Distance from segments (...,2,3) to an AABB; exact up to the line search
        (the distance along a segment to a convex set is convex in t)."""
        a, b = seg[..., 0, :], seg[..., 1, :]

        ab = b - a

        def dist2(t):
            p = a + t[..., None] * ab
            d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
            return np.einsum("...i,...i->...", d, d)

        lo_t = np.zeros(a.shape[:-1])
        hi_t = np.ones(a.shape[:-1])
        g = (np.sqrt(5) - 1) / 2
        for _ in range(iters):
            m1 = hi_t - g * (hi_t - lo_t)
            m2 = lo_t + g * (hi_t - lo_t)
            f1, f2 = dist2(m1), dist2(m2)
            left = f1 < f2
            hi_t = np.where(left, m2, hi_t)
            lo_t = np.where(left, lo_t, m1)
        t = 0.5 * (lo_t + hi_t)
        d2 = np.minimum(np.minimum(dist2(t), dist2(np.zeros_like(t))), dist2(np.ones_like(t)))
        return np.sqrt(d2)

    def clearance(self, q, return_detail=False):
        """Signed clearance margin (mm) per configuration: min over primitives and
        obstacles of (distance - radius - required clearance). >= 0 is collision-free."""
        q = np.asarray(q, float).reshape(-1, 6)
        caps = self.capsules(q)                      # (N, M, 2, 3)
        out = np.full(q.shape[0], np.inf)
        detail = {}
        tip = np.array([n == "tool_tip" for n in self.names])
        for box in self.boxes:
            # cheap lower bound: distance between the segment's AABB and the box
            gap = np.maximum(np.maximum(box.lo - caps.max(-2), caps.min(-2) - box.hi), 0.0)
            lower = np.linalg.norm(gap, axis=-1) - self.radii - box.required_clearance
            margin = lower.copy()
            near = lower < 50.0
            if near.any():
                d = self.segment_box_distance(caps[near], box.lo, box.hi)
                margin[near] = d - self.radii[np.nonzero(near)[1]] - box.required_clearance
            if box.name == "bed":
                margin[:, tip] = np.inf
            m = margin.min(axis=1)
            detail[box.name] = m
            out = np.minimum(out, m)
        if np.isfinite(self.floor_z):
            zmin = caps[..., 2].min(-1) - self.radii
            zmin[:, tip] = np.inf
            m = (zmin - self.floor_z).min(axis=1)
            detail["floor"] = m
            out = np.minimum(out, m)
        return (out, detail) if return_detail else out

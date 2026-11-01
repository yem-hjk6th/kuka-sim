"""Procedural non-planar test toolpaths over the bed.

A surface z = f(x, y) is defined over a square patch. The planned layer is
z = bed_top + base_height + f + layer_index * layer_height. The path is a
boustrophedon raster (lines along X), densified to ds mm by 3D arc length.
The nozzle direction (tool +X) is minus the surface normal, tilted at most
max_tilt from straight down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Toolpath:
    name: str
    points: np.ndarray       # (N, 3) mm, BASE 0
    dirs: np.ndarray         # (N, 3) unit nozzle direction (tool +X), pointing into the print
    tilt_deg: np.ndarray     # (N,) nozzle tilt from vertical actually used
    normal_tilt_deg: np.ndarray  # (N,) surface normal tilt (before the cap)
    s: np.ndarray            # (N,) arc length mm
    max_tilt_deg: float


def _softplus(u, k):
    return np.logaddexp(0.0, k * u) / k


def surface_funcs(a):
    """Height functions f(u, v) over the patch [-a, a]^2 (local coordinates)."""
    H_dome = 80.0

    def dome(u, v):
        # smooth bump H (1 - (r/a)^2)^2: tangent to the bed at r = a, steepest 46 deg at r = a/sqrt(3)
        w = np.maximum(0.0, 1 - (u * u + v * v) / (a * a))
        return H_dome * w * w

    def saddle(u, v):
        return 40.0 * (u * u - v * v) / (a * a) + 40.0

    def sine(u, v):
        return 15.0 * np.sin(2 * np.pi * u / 120.0) + 15.0

    def ramp45(u, v):
        # flat - 45 deg ramp rising 60 mm - flat, with ~10 mm fillets
        k = 0.15
        return _softplus(u + 30.0, k) - _softplus(u - 30.0, k)

    def flat(u, v):
        return np.zeros_like(u)

    return {"dome": dome, "saddle": saddle, "sine_wave": sine, "ramp_45": ramp45, "flat": flat}


def cap_tilt(normal, max_tilt_deg):
    """Nozzle direction = -normal, rotated towards -Z so that tilt <= max_tilt."""
    d = -normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    down = np.array([0, 0, -1.0])
    tilt = np.arccos(np.clip(d @ down, -1, 1))
    mt = np.radians(max_tilt_deg)
    # horizontal component direction
    h = d - (d @ down)[:, None] * down
    hn = np.linalg.norm(h, axis=-1, keepdims=True)
    h = np.where(hn > 1e-12, h / np.maximum(hn, 1e-12), 0.0)
    t = np.minimum(tilt, mt)
    out = np.cos(t)[:, None] * down + np.sin(t)[:, None] * h
    return out, np.degrees(t), np.degrees(tilt)


def _resample(xy_poly, zfun, ds):
    # fine sampling of the xy polyline
    seg = np.diff(xy_poly, axis=0)
    pts = [xy_poly[:1]]
    for p0, d in zip(xy_poly[:-1], seg):
        n = max(2, int(np.ceil(np.linalg.norm(d) / 0.1)))
        t = np.linspace(0, 1, n)[1:, None]
        pts.append(p0 + t * d)
    xy = np.concatenate(pts)
    z = zfun(xy[:, 0], xy[:, 1])
    P = np.column_stack([xy, z])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = int(np.floor(s[-1] / ds)) + 1
    si = np.linspace(0, s[-1], n)
    xi = np.interp(si, s, xy[:, 0])
    yi = np.interp(si, s, xy[:, 1])
    return xi, yi, si


def make_toolpath(cfg, name, max_tilt_deg=None):
    tp = cfg.planner["toolpath"]
    ds = float(tp["ds_mm"])
    max_tilt = float(tp["max_tilt_deg"] if max_tilt_deg is None else max_tilt_deg)
    bed = cfg.bed()
    z0 = bed["top_z"] + float(tp["base_height"]) + int(tp["layer_index"]) * cfg.layer_height

    if name == "perimeter":
        ins = float(tp["perimeter_inset"])
        x0, x1 = bed["x"][0] + ins, bed["x"][1] - ins
        y0, y1 = bed["y"][0] + ins, bed["y"][1] - ins
        poly = np.array([[x0, y1], [x1, y1], [x1, y0], [x0, y0], [x0, y1]])
        z0p = bed["top_z"] + cfg.layer_height      # first layer on the bare bed
        f_world = lambda x, y: np.full_like(x, z0p)
    else:
        cx, cy = tp["patch_center_xy"]
        a = float(tp["patch_half_size"])
        f = surface_funcs(a)[name]
        f_world = lambda x, y: z0 + f(x - cx, y - cy)
        sp = float(tp["raster_spacing"])
        ys = np.arange(-a, a + 1e-9, sp)
        poly = []
        for i, v in enumerate(ys):
            xs = (-a, a) if i % 2 == 0 else (a, -a)
            poly += [[cx + xs[0], cy + v], [cx + xs[1], cy + v]]
        poly = np.array(poly)

    xi, yi, si = _resample(poly, f_world, ds)
    zi = f_world(xi, yi)
    h = 0.01
    fx = (f_world(xi + h, yi) - f_world(xi - h, yi)) / (2 * h)
    fy = (f_world(xi, yi + h) - f_world(xi, yi - h)) / (2 * h)
    normal = np.column_stack([-fx, -fy, np.ones_like(fx)])
    dirs, tilt, ntilt = cap_tilt(normal, max_tilt)
    return Toolpath(name, np.column_stack([xi, yi, zi]), dirs, tilt, ntilt, si, max_tilt)


SURFACES = ["dome", "saddle", "sine_wave", "ramp_45", "perimeter"]

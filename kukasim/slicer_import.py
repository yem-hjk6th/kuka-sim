"""Optional reader for KRL .src files produced by non_planar_slicer (GPL-3.0).

Kept in its own module on purpose: it only parses the slicer's *output* (LIN
targets with X Y Z A B C) and contains no code from that project. The nozzle
direction is taken as tool +X of each pose; the slicer's own rotation about it
(C = 0) is discarded, because choosing it is the planner's job.
"""
from __future__ import annotations

import re

import numpy as np

from .rotations import abc_to_matrix
from .toolpaths import Toolpath

_NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
_LIN = re.compile(r"\b(?:LIN|SLIN|PTP)\s*\{([^}]*)\}", re.IGNORECASE)


def parse_src(text):
    poses = []
    for m in _LIN.finditer(text):
        body = m.group(1)
        vals = {}
        for k in ("X", "Y", "Z", "A", "B", "C"):
            mm = re.search(rf"\b{k}\s+{_NUM}", body)
            if mm:
                vals[k] = float(mm.group(1))
        if all(k in vals for k in ("X", "Y", "Z", "A", "B", "C")):
            poses.append([vals[k] for k in ("X", "Y", "Z", "A", "B", "C")])
    return np.array(poses).reshape(-1, 6)


def densify(points, dirs, ds):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    n = max(2, int(np.floor(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    P = np.column_stack([np.interp(si, s, points[:, k]) for k in range(3)])
    D = np.column_stack([np.interp(si, s, dirs[:, k]) for k in range(3)])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return P, D, si


def toolpath_from_src(path, ds=1.5, name=None):
    with open(path) as f:
        poses = parse_src(f.read())
    if len(poses) < 2:
        raise ValueError(f"{path}: fewer than two Cartesian targets")
    R = abc_to_matrix(poses[:, 3], poses[:, 4], poses[:, 5])
    dirs = R[:, :, 0]
    P, D, s = densify(poses[:, :3], dirs, ds)
    tilt = np.degrees(np.arccos(np.clip(-D[:, 2], -1, 1)))
    return Toolpath(name or str(path), P, D, tilt, tilt, s, float(tilt.max()))

"""Rotation helpers.

Everything inside the package works with 3x3 rotation matrices (and 4x4
homogeneous transforms, positions in mm). KUKA A/B/C angles are only produced
or consumed at the boundary (CSV, KRL, validation input).

KUKA ABC convention: R = Rz(A) @ Ry(B) @ Rx(C)  (A about Z, then B about the
new Y, then C about the new X). B = +-90 deg is the Euler singularity: there
only A - C (B = +90) or A + C (B = -90) is defined.
"""
from __future__ import annotations

import numpy as np


def rx(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1),
                     np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def ry(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1),
                     np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def rz(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1),
                     np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def rpy_to_matrix(r, p, y):
    """URDF rpy: R = Rz(y) Ry(p) Rx(r)."""
    return rz(y) @ ry(p) @ rx(r)


def axis_angle(axis, angle):
    """Rodrigues rotation(s) about a unit axis (3,), angle scalar or (N,)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.asarray(angle, dtype=float)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    s = np.sin(angle)[..., None, None]
    c = np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def abc_to_matrix(a_deg, b_deg, c_deg):
    a, b, c = np.radians(a_deg), np.radians(b_deg), np.radians(c_deg)
    return rz(a) @ ry(b) @ rx(c)


def matrix_to_abc(R, c_hint_deg=None, sing_tol=1e-6):
    """Rotation matrix -> KUKA (A, B, C) in degrees.

    B in [-90, 90], A and C in (-180, 180]. At the Euler singularity
    (|cos B| < sing_tol) the split between A and C is arbitrary; C is set to
    ``c_hint_deg`` (e.g. the previous point's C) or 0, so that a sequence of
    poses through B = 90 stays continuous.
    """
    R = np.asarray(R, dtype=float)
    sb = -R[2, 0]
    sb = min(1.0, max(-1.0, sb))
    cb = np.hypot(R[0, 0], R[1, 0])
    if cb > sing_tol:
        b = np.arctan2(sb, cb)
        a = np.arctan2(R[1, 0], R[0, 0])
        c = np.arctan2(R[2, 1], R[2, 2])
    else:
        c = np.radians(c_hint_deg) if c_hint_deg is not None else 0.0
        if sb > 0:   # B = +90: R = Rz(A - C) Ry(90)
            b = np.pi / 2
            psi = np.arctan2(-R[0, 1], R[1, 1])
            a = psi + c
        else:        # B = -90: R = Rz(A + C) Ry(-90)
            b = -np.pi / 2
            psi = np.arctan2(-R[0, 1], R[1, 1])
            a = psi - c
        a = np.arctan2(np.sin(a), np.cos(a))
    return float(np.degrees(a)), float(np.degrees(b)), float(np.degrees(c))


def rotation_angle_between(R1, R2):
    """Geodesic angle (rad) between rotation matrices; broadcasts over (...,3,3)."""
    Rr = np.swapaxes(R1, -1, -2) @ R2
    tr = np.trace(Rr, axis1=-2, axis2=-1)
    v = np.stack([Rr[..., 2, 1] - Rr[..., 1, 2], Rr[..., 0, 2] - Rr[..., 2, 0],
                  Rr[..., 1, 0] - Rr[..., 0, 1]], -1)
    # atan2 form: accurate for small angles (arccos of the trace is not)
    return np.arctan2(np.linalg.norm(v, axis=-1), tr - 1)


def make_transform(R, p):
    T = np.zeros(np.broadcast_shapes(np.shape(R)[:-2], np.shape(p)[:-1]) + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

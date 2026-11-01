"""FK and analytic IK of the KR 10 R1100-2 from its URDF, with TOOL and BASE frames.

* Lengths in mm, joint angles in rad inside the package (degrees at the boundary).
* URDF joint angles equal KUKA A1..A6 (the URDF axis signs -Z/+Y/+Y/-X/+Y/-Z
  follow KUKA's directions; checked by the one logged pose noted in cell.yaml).
* The URDF link_6 frame equals KUKA's FLANGE frame (Z out of the flange,
  X pointing down at A1..A6 = 0,-90,90,0,0,0, i.e. FLANGE ABC = 0, 90, 0).
* The tool (TOOL 3) is applied as FLANGE @ T_tool; KUKA tool direction is +X,
  so the nozzle axis is tool +X.

IK returns every configuration: 2 (shoulder / overhead) x 2 (elbow) x 2 (wrist
flip) x 2 (A4 turn) x 2 (A6 turn) = 32 slots, with a validity mask. A1, A2, A3
and A5 have ranges narrower than 360 deg, so at most one representation is in
range; A4 (+-185) and A6 (+-350) can have two, which is the turn ambiguity and
is enumerated explicitly.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .rotations import (abc_to_matrix, axis_angle, make_transform, matrix_to_abc,
                        rpy_to_matrix, wrap_pi)

N_SLOTS = 32


@dataclass
class Joint:
    name: str
    origin_xyz: np.ndarray   # mm
    origin_rpy: np.ndarray   # rad
    axis: np.ndarray
    lower: float             # rad
    upper: float
    velocity: float          # rad/s

    @property
    def T_origin(self):
        return make_transform(rpy_to_matrix(*self.origin_rpy), self.origin_xyz)


def _snap_right_angles(rpy, tol=1e-4):
    """SW2URDF writes pi/2 as 1.5708 (3.7e-6 rad off). Snap such values to exact
    multiples of pi/2 so FK and the analytic IK describe the same robot."""
    k = np.round(rpy / (np.pi / 2))
    return np.where(np.abs(rpy - k * np.pi / 2) < tol, k * np.pi / 2, rpy)


def parse_urdf(path, snap=True) -> list[Joint]:
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        o = j.find("origin")
        lim = j.find("limit")
        joints.append(Joint(
            name=j.get("name"),
            origin_xyz=np.array([float(v) for v in o.get("xyz").split()]) * 1000.0,
            origin_rpy=(_snap_right_angles if snap else (lambda r: r))(
                np.array([float(v) for v in o.get("rpy").split()])),
            axis=np.array([float(v) for v in j.find("axis").get("xyz").split()]),
            lower=float(lim.get("lower")), upper=float(lim.get("upper")),
            velocity=float(lim.get("velocity")),
        ))
    joints.sort(key=lambda jj: jj.name)
    assert [j.name for j in joints] == [f"joint_{i}" for i in range(1, 7)]
    return joints


def xyzabc_to_T(xyzabc):
    x, y, z, a, b, c = xyzabc
    return make_transform(abc_to_matrix(a, b, c), np.array([x, y, z], float))


def T_to_xyzabc(T, c_hint_deg=None):
    a, b, c = matrix_to_abc(T[:3, :3], c_hint_deg=c_hint_deg)
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), a, b, c]


class Robot:
    def __init__(self, urdf_path, tool_xyzabc=(0, 0, 0, 0, 0, 0),
                 base_xyzabc=(0, 0, 0, 0, 0, 0), limits_deg=None):
        self.joints = parse_urdf(urdf_path)
        self.T_tool = xyzabc_to_T(tool_xyzabc)
        self.T_tool_inv = np.linalg.inv(self.T_tool)
        self.T_base = xyzabc_to_T(base_xyzabc)
        self.T_base_inv = np.linalg.inv(self.T_base)
        if limits_deg is None:
            limits_deg = [(np.degrees(j.lower), np.degrees(j.upper)) for j in self.joints]
        self.limits = np.radians(np.array(limits_deg, float))   # (6, 2)
        self.vmax = np.array([j.velocity for j in self.joints])  # rad/s
        self._check_structure()

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.urdf_path, cfg.tool_xyzabc, cfg.base_xyzabc, cfg.joint_limits_deg)

    # ------------------------------------------------------------------ FK
    def link_frames(self, q):
        """q (...,6) rad -> list of 6 transforms (...,4,4) of link_1..link_6 in the robot root."""
        q = np.asarray(q, float)
        T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        frames = []
        for i, j in enumerate(self.joints):
            Rq = axis_angle(j.axis, q[..., i])
            T = T @ j.T_origin @ make_transform(Rq, np.zeros(q.shape[:-1] + (3,)))
            frames.append(T)
        return frames

    def fk_flange(self, q):
        """Flange pose in the robot root (= world)."""
        return self.link_frames(q)[-1]

    def fk(self, q):
        """TCP pose (TOOL) expressed in BASE."""
        return self.T_base_inv @ self.fk_flange(q) @ self.T_tool

    def fk_xyzabc(self, q_deg):
        return T_to_xyzabc(self.fk(np.radians(q_deg)))

    # ----------------------------------------------------------- geometry
    def _check_structure(self):
        J = self.joints
        exp_axes = [(0, 0, -1), (0, 1, 0), (0, 1, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1)]
        for j, a in zip(J, exp_axes):
            if not np.allclose(j.axis, a):
                raise ValueError(f"{j.name}: axis {j.axis} differs from the analytic-IK model {a}")
        for j in J[:5]:
            if not np.allclose(j.origin_rpy, 0):
                raise ValueError(f"{j.name}: unexpected origin rpy")
        if not np.allclose(J[5].origin_rpy, [0, np.pi / 2, 0], atol=1e-4):
            raise ValueError("joint_6 origin rpy is not (0, pi/2, 0)")
        # Lateral (Y) offsets must cancel so the wrist centre lies in the arm plane.
        y_sum = sum(j.origin_xyz[1] for j in J[1:6])
        if abs(y_sum) > 1e-6 or abs(J[0].origin_xyz[1]) > 1e-9:
            raise ValueError("lateral offsets do not cancel")
        self.d1 = J[0].origin_xyz[2] + J[1].origin_xyz[2]   # shoulder height
        self.a1 = J[0].origin_xyz[0] + J[1].origin_xyz[0]   # shoulder offset
        self.a2 = J[2].origin_xyz[0]                         # upper arm
        if abs(J[2].origin_xyz[2]) > 1e-9:
            raise ValueError("joint_3 z offset not supported")
        fx = J[3].origin_xyz[0] + J[4].origin_xyz[0]        # A3 -> wrist centre, along link_3 x
        fz = J[3].origin_xyz[2] + J[4].origin_xyz[2]        # ... and z
        self.a3 = float(np.hypot(fx, fz))
        self.delta3 = float(np.arctan2(fz, fx))
        self.d6 = J[5].origin_xyz[0]                         # wrist centre -> flange along A6
        if abs(J[5].origin_xyz[2]) > 1e-9:
            raise ValueError("joint_6 z offset not supported")

    def wrist_center(self, T_flange):
        return T_flange[..., :3, 3] - self.d6 * T_flange[..., :3, 2]

    # ------------------------------------------------------------------ IK
    def ik_flange(self, T_flange, limits=None):
        """All IK solutions for flange poses (N,4,4) in the robot root.

        Returns q (N, 32, 6) rad and valid (N, 32) bool (reachable and inside
        ``limits`` (6,2) rad, default: the configured joint limits).
        """
        T = np.asarray(T_flange, float).reshape(-1, 4, 4)
        lim = self.limits if limits is None else np.asarray(limits, float)
        N = T.shape[0]
        R = T[:, :3, :3]
        wc = self.wrist_center(T)
        q = np.full((N, 2, 2, 2, 2, 2, 6), np.nan)
        ok = np.zeros((N, 2, 2, 2, 2, 2), bool)
        q1_front = np.arctan2(-wc[:, 1], wc[:, 0])
        rho = np.hypot(wc[:, 0], wc[:, 1])
        for s in range(2):
            q1 = q1_front + (np.pi if s else 0.0)
            x1 = -rho if s else rho
            u = x1 - self.a1
            v = -(wc[:, 2] - self.d1)
            c2 = (u * u + v * v - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)
            reach = np.abs(c2) <= 1.0
            c2 = np.clip(c2, -1, 1)
            for e in range(2):
                t2 = np.arccos(c2) * (1 if e == 0 else -1)
                t1 = np.arctan2(v, u) - np.arctan2(self.a3 * np.sin(t2), self.a2 + self.a3 * np.cos(t2))
                q2, q3 = t1, t2 + self.delta3
                # R03 = Rz(-q1) Ry(q2 + q3)
                R03 = axis_angle((0, 0, -1), q1) @ axis_angle((0, 1, 0), q2 + q3)
                M = np.swapaxes(R03, -1, -2) @ R      # = Rx(-q4) Ry(q5 + pi/2) Rz(-q6)
                sb = np.clip(M[:, 0, 2], -1, 1)
                cb_abs = np.sqrt(np.maximum(0.0, 1 - sb * sb))
                for w in range(2):
                    cb = cb_abs * (1 if w == 0 else -1)
                    sing = cb_abs < 1e-9
                    cbs = np.where(sing, 1.0, cb)
                    a = np.arctan2(-M[:, 1, 2] / cbs, M[:, 2, 2] / cbs)
                    c = np.arctan2(-M[:, 0, 1] / cbs, M[:, 0, 0] / cbs)
                    # At the wrist singularity only a + c (sb > 0) or c - a (sb < 0)
                    # is defined; then M = Ry(+-90) Rz(phi) and A4 is set to 0.
                    a = np.where(sing, 0.0, a)
                    c = np.where(sing, np.arctan2(M[:, 1, 0], M[:, 1, 1]), c)
                    b = np.arctan2(sb, cb)
                    q4, q5, q6 = -a, b - np.pi / 2, -c
                    base = np.stack([q1, q2, q3, q4, q5, q6], -1)
                    base = wrap_pi(base)
                    # single-representation joints: pick the one in range
                    fixed = base.copy()
                    in_lim = np.ones(N, bool)
                    for k in (0, 1, 2, 4):
                        cand = np.stack([base[:, k], base[:, k] - 2 * np.pi, base[:, k] + 2 * np.pi], -1)
                        inr = (cand >= lim[k, 0] - 1e-12) & (cand <= lim[k, 1] + 1e-12)
                        idx = np.argmax(inr, axis=1)
                        fixed[:, k] = cand[np.arange(N), idx]
                        in_lim &= inr.any(axis=1)
                    for t4 in range(2):
                        for t6 in range(2):
                            qq = fixed.copy()
                            for k, t in ((3, t4), (5, t6)):
                                if t:
                                    sgn = np.where(qq[:, k] >= 0, 1.0, -1.0)
                                    qq[:, k] = qq[:, k] - 2 * np.pi * sgn
                            v_ok = in_lim & reach
                            v_ok &= (qq[:, 3] >= lim[3, 0]) & (qq[:, 3] <= lim[3, 1])
                            v_ok &= (qq[:, 5] >= lim[5, 0]) & (qq[:, 5] <= lim[5, 1])
                            q[:, s, e, w, t4, t6] = qq
                            ok[:, s, e, w, t4, t6] = v_ok
        return q.reshape(N, N_SLOTS, 6), ok.reshape(N, N_SLOTS)

    def ik(self, T_tcp_in_base, limits=None):
        """IK for TCP poses given in BASE (TOOL applied)."""
        T = np.asarray(T_tcp_in_base, float).reshape(-1, 4, 4)
        T_flange = self.T_base @ T @ self.T_tool_inv
        return self.ik_flange(T_flange, limits)

    # --------------------------------------------------- KUKA S / T bits
    def turn_bits(self, q):
        """KUKA Turn: bit i set if A(i+1) < 0."""
        q = np.asarray(q)
        return sum((1 << i) * (q[..., i] < 0).astype(int) for i in range(6))

    def status_bits(self, q):
        """KUKA Status (bits 0-2). UNVERIFIED against the controller: bit 0 =
        wrist centre behind the A1 axis (overhead), bit 1 = A3 >= phi where phi
        is the A3 angle at which A3, the wrist centre and A2 are in line, bit 2
        = A5 <= 0. Use the joint values (CSV / E6AXIS) as the reference."""
        q = np.asarray(q, float)
        T6 = self.fk_flange(q)
        wc = self.wrist_center(T6)
        x1 = wc[..., 0] * np.cos(q[..., 0]) - wc[..., 1] * np.sin(q[..., 0])
        b0 = (x1 < 0).astype(int)
        b1 = (q[..., 2] >= self.delta3).astype(int)
        b2 = (q[..., 4] <= 0).astype(int)
        return b0 + 2 * b1 + 4 * b2

"""Per-toolpath outputs: trajectory CSV, plots, feasibility map, animation, KRL export.

Numbers in the CSV use the fixed form of kukasim/krl.py (fnum / fang): fixed decimals,
never a negative zero, orientation angles A / C in (-180, 180]."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .krl import fang, fnum, write_krl  # noqa: E402,F401  (write_krl: re-export)
from .rotations import matrix_to_abc  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
INFEAS = "#e34948"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
    "axes.titlesize": 9, "axes.titlecolor": INK, "lines.linewidth": 1.5,
})


def tcp_rows(robot, res):
    """TCP pose and ABC per planned point (continuous C through B = 90)."""
    rows = []
    c_prev = 0.0
    for i in range(len(res.bins)):
        if np.isnan(res.q[i, 0]):
            rows.append(None)
            continue
        T = robot.fk(res.q[i])
        a, b, c = matrix_to_abc(T[:3, :3], c_hint_deg=c_prev)
        c_prev = c
        rows.append((T[:3, 3], (a, b, c)))
    return rows


def unwrap_deg(x):
    out = np.array(x, float)
    ok = ~np.isnan(out)
    out[ok] = np.degrees(np.unwrap(np.radians(out[ok])))
    return out


def segment_ids(res):
    seg = np.full(len(res.bins), -1)
    for k, (a, b) in enumerate(res.segments):
        seg[a:b + 1] = k
    return seg


def write_csv(path, robot, res, rows):
    tp = res.toolpath
    seg = segment_ids(res)
    qd = np.degrees(res.q)
    S = robot.status_bits(np.nan_to_num(res.q))
    T = robot.turn_bits(np.nan_to_num(res.q))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "s_mm", "x", "y", "z", "dir_x", "dir_y", "dir_z", "tilt_deg", "feasible",
                    "segment", "theta_deg", "ik_slot", "A1", "A2", "A3", "A4", "A5", "A6",
                    "X", "Y", "Z", "A", "B", "C", "S_unverified", "T", "limit_margin_deg",
                    "abs_A5_deg", "clearance_mm_capped50", "node_cost"])
        for i in range(len(res.bins)):
            base = [i, fnum(tp.s[i], 3), *[fnum(v, 3) for v in tp.points[i]],
                    *[fnum(v, 6) for v in tp.dirs[i]], fnum(tp.tilt_deg[i], 3)]
            if rows[i] is None:
                w.writerow(base + [0, -1] + [""] * 23)
                continue
            p, abc = rows[i]
            w.writerow(base + [1, seg[i], fnum(res.thetas_deg[res.bins[i]], 1), res.slots[i],
                               *[fnum(v, 4) for v in qd[i]], *[fnum(v, 3) for v in p],
                               fang(abc[0], 4), fnum(abc[1], 4), fang(abc[2], 4), int(S[i]), int(T[i]),
                               fnum(res.limit_margin[i], 3), fnum(abs(qd[i, 4]), 3),
                               fnum(res.clearance[i], 2), fnum(res.node_cost[i], 5)])


def _shade_infeasible(ax, s, feasible):
    bad = ~feasible
    if not bad.any():
        return
    idx = np.nonzero(bad)[0]
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]]
    for a, b in zip(starts, ends):
        ax.axvspan(s[a], s[b], color=INFEAS, alpha=0.15, lw=0)


def plot_joints(path, robot, res, title):
    s = res.toolpath.s
    qd = np.degrees(res.q)
    lim = np.degrees(robot.limits)
    fig, axes = plt.subplots(6, 1, figsize=(7, 8), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(s, qd[:, j], color=BLUE)
        lo, hi = lim[j]
        y0, y1 = np.nanmin(qd[:, j]) if res.feasible.any() else lo, np.nanmax(qd[:, j]) if res.feasible.any() else hi
        for L in (lo, hi):
            if y0 - 40 < L < y1 + 40:
                ax.axhline(L, color=GRAY, ls="--", lw=1)
        ax.set_ylabel(f"A{j + 1} [deg]")
        _shade_infeasible(ax, s, res.feasible)
    axes[0].set_title(f"{title}: joint trajectory (dashed: joint limit, red band: infeasible)", loc="left")
    axes[-1].set_xlabel("path length [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_margins(path, res, cfg, title):
    s = res.toolpath.s
    th = np.where(res.bins >= 0, res.thetas_deg[np.clip(res.bins, 0, None)], np.nan)
    fig, axes = plt.subplots(5, 1, figsize=(7, 7.5), sharex=True)
    data = [
        (unwrap_deg(th), "redundancy angle [deg]", None),
        (res.limit_margin, "joint-limit margin [deg]\n(beyond the 2 deg margin)", 0),
        (np.abs(np.degrees(res.q[:, 4])), "|A5| [deg]", cfg.planner["planner"]["a5_soft_zone_deg"]),
        (res.clearance, "clearance [mm]\n(capped at 50)", 0),
        (res.toolpath.tilt_deg, "nozzle tilt [deg]", None),
    ]
    for ax, (y, lab, ref) in zip(axes, data):
        ax.plot(s, y, color=BLUE)
        if ref is not None:
            ax.axhline(ref, color=GRAY, ls="--", lw=1)
        ax.set_ylabel(lab)
        _shade_infeasible(ax, s, res.feasible)
    axes[-1].plot(s, res.toolpath.normal_tilt_deg, color=ORANGE, lw=1)
    axes[-1].text(0.99, 0.95, "blue: used   orange: surface normal", transform=axes[-1].transAxes,
                  ha="right", va="top", color=MUTED)
    axes[0].set_title(f"{title}: chosen redundancy angle and margins", loc="left")
    axes[-1].set_xlabel("path length [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_feasibility(path, res, title):
    F = res.feas_count.T.astype(float)
    s = res.toolpath.s
    fig, ax = plt.subplots(figsize=(7, 3.6))
    F[F == 0] = np.nan
    th = res.thetas_deg
    step = th[1] - th[0]
    im = ax.imshow(F, aspect="auto", origin="lower", cmap="Blues", vmin=0,
                   extent=[s[0], s[-1], th[0] - step / 2, th[-1] + step / 2],
                   interpolation="nearest")
    ax.set_facecolor("#f1efe8")
    chosen = np.where(res.bins >= 0, th[np.clip(res.bins, 0, None)], np.nan)
    ax.plot(s, chosen, ".", ms=1.5, color=ORANGE)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("feasible IK configs")
    ax.set_xlabel("path length [mm]")
    ax.set_ylabel("redundancy angle [deg]")
    ax.set_title(f"{title}: feasibility map (beige: none feasible, orange: chosen)", loc="left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _box_edges(lo, hi):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in e]


def animate(path, coll, res, title, n_frames=36):
    """Stick-figure animation of the planned motion (GIF via Pillow)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    idx = np.nonzero(res.feasible)[0]
    if len(idx) == 0:
        return False
    frames = idx[np.linspace(0, len(idx) - 1, min(n_frames, len(idx))).astype(int)]
    caps = coll.capsules(res.q[frames])
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(projection="3d")
    pts = res.toolpath.points

    def draw(k):
        ax.cla()
        for box in coll.boxes:
            for a, b in _box_edges(box.lo, box.hi):
                ax.plot(*zip(a, b), color=GRAY, lw=0.6)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=ORANGE, lw=0.6)
        done = pts[: frames[k] + 1]
        ax.plot(done[:, 0], done[:, 1], done[:, 2], color=INK, lw=1.0)
        for m, seg in enumerate(caps[k]):
            col = BLUE if not coll.names[m].startswith("tool") else ORANGE
            ax.plot(*seg.T, color=col, lw=2 + coll.radii[m] / 20, solid_capstyle="round")
        ax.set_xlim(-100, 1100)
        ax.set_ylim(-900, 300)
        ax.set_zlim(-50, 1150)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"{title}  point {frames[k]}", fontsize=8)
        ax.tick_params(labelsize=6)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=6), dpi=70)
    plt.close(fig)
    return True


def plot_turn_window(path, xs, ys, W, cfg):
    """Small multiples: continuous turn about the nozzle axis over the bed."""
    keys = list(W)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys) + 0.8, 2.9), sharey=True)
    vmax = max(360.0, max(v.max() for v in W.values()))
    for ax, k in zip(axes, keys):
        im = ax.imshow(W[k], origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                       extent=[xs[0] - 20, xs[-1] + 20, ys[0] - 20, ys[-1] + 20])
        for ob in cfg.cell.get("obstacles", []):
            if "pad_xy" in ob:
                ax.plot(*ob["pad_xy"], "x", color=ORANGE, ms=6)
        ax.set_title(k.replace("_", " "), fontsize=8)
        ax.set_xlabel("X [mm]")
        ax.grid(False)
    axes[0].set_ylabel("Y [mm]")
    cb = fig.colorbar(im, ax=axes, pad=0.01, shrink=0.9)
    cb.set_label("continuous turn [deg]")
    fig.suptitle("Continuous rotation about the nozzle axis, tip 10-30 mm above the bed (x: clamp)",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)

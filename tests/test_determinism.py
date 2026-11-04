"""The DP plan must not depend on the platform (P4-2): ties are broken deterministically,
and a fresh plan reproduces every committed results/<surface>/trajectory.csv."""
import csv
from pathlib import Path

import numpy as np
import pytest

from kukasim.planner import COST_TIE_RTOL, _tie_break
from kukasim.toolpaths import SURFACES

REPO = Path(__file__).resolve().parent.parent


def _committed_q_deg(name):
    with open(REPO / "results" / name / "trajectory.csv") as f:
        return np.array([[float(r[f"A{j}"] or "nan") for j in range(1, 7)] for r in csv.DictReader(f)])


@pytest.mark.parametrize("name", SURFACES)
def test_fresh_plan_equals_committed_trajectory(fresh_plans, name):
    _, _, _, plans = fresh_plans
    q = np.degrees(plans[name].q)
    old = _committed_q_deg(name)
    assert q.shape == old.shape
    assert (np.isnan(q) == np.isnan(old)).all()
    assert np.allclose(np.nan_to_num(q), np.nan_to_num(old), atol=1e-3), \
        f"{name}: fresh plan differs from results/{name}/trajectory.csv by " \
        f"{np.nanmax(np.abs(q - old)):.3f} deg"


def test_tie_break_is_independent_of_float_noise_and_order():
    rng = np.random.default_rng(0)
    q = np.zeros((4, 6))
    q[:, 3] = np.radians([180, 0, 0, -10])      # |A4| + |A6|: 180+0, 0+360, 0+0, 10+0
    q[:, 5] = np.radians([0, 360, 0, 0])
    cost = np.array([5.0, 5.0, 5.0, 5.0]) + rng.uniform(-1, 1, 4) * COST_TIE_RTOL * 0.4
    group = np.zeros(4, int)
    idx = np.arange(4)
    assert _tie_break(group, cost, q, idx).tolist() == [2]          # smallest |A4| + |A6|
    # whatever the noise and the order, the same state wins
    for _ in range(20):
        p = rng.permutation(4)
        noisy = cost[p] + rng.uniform(-1, 1, 4) * COST_TIE_RTOL * 0.4
        assert p[_tie_break(group, noisy, q[p], idx[p])].tolist() == [2]
    # equal wrist too: lowest index
    q2 = np.zeros((3, 6))
    assert _tie_break(np.zeros(3, int), np.array([1.0, 1.0, 1.0]), q2, np.array([7, 3, 5])).tolist() == [1]
    # a real cost difference still wins over the wrist rule
    assert _tie_break(group, np.array([5.0, 5.0, 5.0 + 1e-3, 5.0]), q, idx).tolist() == [3]
    # per group
    g = np.array([0, 0, 1, 1])
    assert _tie_break(g, np.array([2.0, 1.0, 1.0, 1.0]), q, idx).tolist() == [1, 2]

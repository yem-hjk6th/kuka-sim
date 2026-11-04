"""Shared fixtures: fresh plans of the committed example toolpaths (planned once per session)."""
import pytest

from kukasim.config import load_config
from kukasim.kinematics import Robot
from kukasim.planner import Planner
from kukasim.toolpaths import SURFACES, make_toolpath


@pytest.fixture(scope="session")
def fresh_plans():
    """(cfg, robot, planner, {surface: PlanResult}) at the default tilt, as run_step1.py plans them."""
    cfg = load_config()
    robot = Robot.from_config(cfg)
    pl = Planner(robot, cfg)
    tilt0 = cfg.planner["toolpath"]["max_tilt_deg"]
    return cfg, robot, pl, {name: pl.plan(make_toolpath(cfg, name, tilt0)) for name in SURFACES}

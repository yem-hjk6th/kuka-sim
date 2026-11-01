"""Loading of config/cell.yaml and config/planner.yaml.

The cell file is read as-is. Values with status ``estimated`` or ``unknown`` are
exposed as ordinary parameters (they can be overridden by the caller), never
hard-coded in the code.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CELL_YAML = REPO / "config" / "cell.yaml"
PLANNER_YAML = REPO / "config" / "planner.yaml"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class Config:
    cell: dict
    planner: dict
    overrides: dict = field(default_factory=dict)

    @property
    def urdf_path(self) -> Path:
        return REPO / self.cell["robot"]["urdf"]

    @property
    def tool_xyzabc(self):
        t = self.cell["tool"]
        return [t["x"], t["y"], t["z"], t["a"], t["b"], t["c"]]

    @property
    def base_xyzabc(self):
        b = self.cell["base"]
        return [b["x"], b["y"], b["z"], b["a"], b["b"], b["c"]]

    @property
    def joint_limits_deg(self):
        jl = self.cell["joint_limits_deg"]
        return [tuple(jl[f"a{i}"]) for i in range(1, 7)]

    @property
    def limit_margin_deg(self) -> float:
        return float(self.cell["joint_limits_deg"]["use_margin_deg"])

    @property
    def print_speed(self) -> float:
        return float(self.cell["process"]["print_speed_mm_s"])

    @property
    def layer_height(self) -> float:
        lh = self.cell["process"]["layer_height_mm"]
        return float(lh[0] if isinstance(lh, list) else lh)

    def bed(self):
        b = self.cell["bed"]
        return {
            "x": tuple(b["outline_xy"]["x"]),
            "y": tuple(b["outline_xy"]["y"]),
            "top_z": float(b["top_z"]),
            "thickness": float(b["thickness"]),
            "rotation_about_z_deg": b.get("rotation_about_z_deg"),
        }

    def statuses(self):
        """(path, status) for every value in cell.yaml that carries a status."""
        out = []

        def walk(d, path):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.endswith("status") and isinstance(v, str):
                        out.append((".".join(path + [k]), v))
                    walk(v, path + [k])
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    name = v.get("name", str(i)) if isinstance(v, dict) else str(i)
                    walk(v, path + [name])
        walk(self.cell, [])
        return out


def load_config(cell_overrides: dict | None = None,
                planner_overrides: dict | None = None,
                cell_path: Path | None = None,
                planner_path: Path | None = None) -> Config:
    with open(cell_path or CELL_YAML) as f:
        cell = yaml.safe_load(f)
    with open(planner_path or PLANNER_YAML) as f:
        planner = yaml.safe_load(f)
    cell = _deep_update(copy.deepcopy(cell), cell_overrides or {})
    planner = _deep_update(copy.deepcopy(planner), planner_overrides or {})
    return Config(cell=cell, planner=planner,
                  overrides={"cell": cell_overrides or {}, "planner": planner_overrides or {}})

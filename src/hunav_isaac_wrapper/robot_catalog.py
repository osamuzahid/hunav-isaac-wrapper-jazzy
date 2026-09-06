"""
Lab robot catalog (isaac-social-nav).

Discovers `config/robots/<name>/robot.yaml` and turns it into the spawn dict
TeleopHuNavSim already understands. Isaac is not imported here — the launcher
(`main.py`) can list `--robot` choices before Kit starts.

Swap axes (orthogonal; do not fork the launcher):
  robot   --robot stretch|reachy     this catalog + vendored USD
  world   --world museum|hospital|…  worlds/*.usd
  people  --config <scenario>        scenarios/*.yaml
  planner Nav2 *or* ESC, not both    nav2_*_params.yaml / esc_*.yaml

Upstream CDN robots stay in teleop_hunav_sim.py. A new lab robot is a new folder
with robot.yaml + USD, not a patch to that table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_KIND_TO_DRIVE = {
    "chassis_only": "chassis_only",
    "static": "static",
    "differential": None,  # WheeledRobot branch in teleop
}


def robots_config_root() -> Path:
    """Directory that contains per-robot folders (`stretch/`, `reachy/`, …)."""
    here = Path(__file__).resolve()
    # .../src/hunav_isaac_wrapper/robot_catalog.py → .../src/config/robots
    src_robots = here.parent.parent / "config" / "robots"
    if src_robots.is_dir():
        return src_robots
    cwd_robots = Path.cwd() / "src" / "config" / "robots"
    if cwd_robots.is_dir():
        return cwd_robots
    cwd2 = Path.cwd() / "config" / "robots"
    if cwd2.is_dir():
        return cwd2
    return src_robots


def list_lab_robot_names() -> List[str]:
    """Folder names that contain robot.yaml. Stretch/Reachy first, then others."""
    root = robots_config_root()
    if not root.is_dir():
        return []
    found = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "robot.yaml").is_file()
    }
    preferred = ("stretch", "stretch_wheeled", "reachy")
    names = [n for n in preferred if n in found]
    names.extend(sorted(found - set(preferred)))
    return names


def list_robot_choices(upstream: List[str]) -> List[str]:
    """Upstream launcher names first, then lab YAML robots (no duplicates)."""
    seen = set()
    out: List[str] = []
    for name in list(upstream) + list_lab_robot_names():
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def load_lab_robot_yaml(name: str) -> Optional[Dict[str, Any]]:
    path = robots_config_root() / name / "robot.yaml"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    data["_yaml_path"] = str(path)
    data["_key"] = name
    return data


def lab_robot_descriptions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in list_lab_robot_names():
        spec = load_lab_robot_yaml(name) or {}
        desc = spec.get("description")
        if desc:
            out[name] = str(desc)
    return out


def _spawn_dict_from_yaml(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Shape TeleopHuNavSim already uses (name, usd_package_file, drive, wheels)."""
    kind = spec.get("kind") or spec.get("drive") or "chassis_only"
    spawn: Dict[str, Any] = {
        "name": spec["name"],
        "usd_package_file": spec["usd_package_file"],
        "spawn_z_lift": float(spec.get("spawn_z_lift", 0.0)),
        "hold_non_wheel_dofs": bool(spec.get("hold_non_wheel_dofs", False)),
        "lab_yaml": spec,
    }
    drive = _KIND_TO_DRIVE.get(kind, kind)
    if drive:
        spawn["drive"] = drive
    if kind == "differential" or spec.get("wheel_dof_names"):
        spawn["wheel_dof_names"] = list(spec["wheel_dof_names"])
        spawn["wheel_radius"] = float(spec["wheel_radius"])
        spawn["wheel_base"] = float(spec["wheel_base"])
    if spec.get("variants"):
        spawn["variants"] = spec["variants"]
    if spec.get("articulation_prim"):
        spawn["articulation_prim"] = spec["articulation_prim"]
    if spec.get("expand_instances"):
        spawn["expand_instances"] = spec["expand_instances"]
    if spec.get("park_kinematic"):
        spawn["park_kinematic"] = spec["park_kinematic"]
    return spawn


def load_lab_spawn_configs() -> Dict[str, Dict[str, Any]]:
    """`--robot` key → teleop spawn dict, for every lab robot.yaml."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in list_lab_robot_names():
        spec = load_lab_robot_yaml(name)
        if spec is None:
            continue
        out[name] = _spawn_dict_from_yaml(spec)
    return out

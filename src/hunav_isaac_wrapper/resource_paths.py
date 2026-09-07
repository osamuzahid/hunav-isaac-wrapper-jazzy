"""Optional overlay directories for lab worlds, robots, maps, and crowds.

The clean wrapper tree does not vend CUCR USDs or Stretch/Reachy YAML.
`social-nav run` sets SOCIAL_NAV_* so TeleopHuNavSim loads those from
social-nav-assets and social-nav-platform. Unset env keeps upstream
src/worlds and src/scenarios.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_dir(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def robots_yaml_dir() -> Path | None:
    return env_dir("SOCIAL_NAV_ROBOTS")


def robot_usd_dir() -> Path | None:
    return env_dir("SOCIAL_NAV_ROBOT_USD")


def worlds_dir() -> Path | None:
    return env_dir("SOCIAL_NAV_WORLDS")


def maps_dir() -> Path | None:
    return env_dir("SOCIAL_NAV_MAPS")


def scenarios_dir() -> Path | None:
    return env_dir("SOCIAL_NAV_SCENARIOS")


def resolve_world_usd(map_name: str, fallback_worlds_dir: str) -> str:
    overlay = worlds_dir()
    if overlay is not None:
        cand = overlay / f"{map_name}.usd"
        if cand.is_file():
            return str(cand)
    return os.path.join(fallback_worlds_dir, f"{map_name}.usd")


def resolve_robot_file(filename: str) -> str | None:
    """Resolve ``reachy/reachy.usd`` against overlay robot trees."""
    for root in (robot_usd_dir(), robots_yaml_dir()):
        if root is None:
            continue
        cand = root / filename
        if cand.is_file():
            return str(cand)
    return None

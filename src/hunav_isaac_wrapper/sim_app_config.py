#!/usr/bin/env python3
"""
SimulationApp launch profiles for HuNav Isaac Wrapper.

---------------------------------------------------------------------------
NEW FILE (isaac-social-nav patch) — did not exist in upstream HuNav_isaac_wrapper v2.0.

ORIGINALLY teleop_hunav_sim.py hard-coded:
  CONFIG = {
      "width": 1280, "height": 720, "sync_loads": True,
      "headless": False, "renderer": "RaytracedLighting",
  }

That had to be patched because our laptop is under Isaac 6.0 RAM/VRAM minima;
always launching the full windowed profile thrashes swap. This module keeps the
original settings as profile "default"|"lab" and adds "debug"|"laptop"
(960x540 headless). Also injects Kit --enable for omni.anim.graph.core /
omni.anim.retarget.core (required on Isaac 6.0; runtime enable+update crashed Kit).
---------------------------------------------------------------------------

Resolves laptop/debug vs lab/default CONFIG without importing Isaac Sim,
so callers can inspect settings before starting Kit.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict, Optional

# Canonical profile names. Aliases map onto these.
DEFAULT_PROFILE = "default"
DEBUG_PROFILE = "debug"

PROFILE_ALIASES = {
    "default": DEFAULT_PROFILE,
    "lab": DEFAULT_PROFILE,
    "debug": DEBUG_PROFILE,
    "laptop": DEBUG_PROFILE,
}

# HuNav needs anim graph + retarget; isaacsim.exp.base.python does not enable them.
# Nav2 also needs the ROS 2 bridge (clock + USD OmniGraph publishers).
# Load at Kit startup via extra_args — runtime enable_extension + app.update() has
# crashed OmniGraph on Isaac 6.0.1 (laptop) during Port item 4 E2E.
_HUNAV_EXTRA_ARGS = [
    "--enable",
    "omni.anim.graph.core",
    "--enable",
    "omni.anim.retarget.core",
    "--enable",
    "isaacsim.ros2.bridge",
    # Viewport DrawLabel overlays for HuNav behavior names (operator demos).
    "--enable",
    "omni.graph.visualization.nodes",
]

# Lab / workstation: original wrapper settings (RaytracedLighting @ 1280x720).
_LAB_CONFIG: Dict[str, Any] = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RaytracedLighting",
    "extra_args": list(_HUNAV_EXTRA_ARGS),
}

# Laptop / debug: lighter for RTX 4070 8GB + 16GB RAM; headless for smoke/E2E.
# Keep RaytracedLighting (RTX lidar needs RTX). 960x540 — do not drop res/AA;
# a quieter viewport starves lidar samples on crowd hops.
_DEBUG_CONFIG: Dict[str, Any] = {
    "width": 960,
    "height": 540,
    "sync_loads": True,
    "headless": True,
    "renderer": "RaytracedLighting",
    "extra_args": list(_HUNAV_EXTRA_ARGS),
}

PROFILES: Dict[str, Dict[str, Any]] = {
    DEFAULT_PROFILE: _LAB_CONFIG,
    DEBUG_PROFILE: _DEBUG_CONFIG,
}


def normalize_profile_name(name: Optional[str]) -> str:
    """Map alias → canonical profile; raise ValueError on unknown names."""
    if name is None or str(name).strip() == "":
        return DEFAULT_PROFILE
    key = str(name).strip().lower()
    if key not in PROFILE_ALIASES:
        known = ", ".join(sorted(PROFILE_ALIASES))
        raise ValueError(f"Unknown HUNAV_ISAAC_PROFILE '{name}'. Use one of: {known}")
    return PROFILE_ALIASES[key]


def _truthy_env(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _profile_from_argv(argv: Optional[list] = None) -> Optional[str]:
    """Best-effort extract --profile / --debug / --laptop from argv (no SystemExit)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--debug" in args:
        return DEBUG_PROFILE
    if "--laptop" in args:
        return DEBUG_PROFILE
    if "--profile" in args:
        i = args.index("--profile")
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            return args[i + 1]
    for a in args:
        if a.startswith("--profile="):
            return a.split("=", 1)[1]
    return None


def _headless_from_argv(argv: Optional[list] = None) -> Optional[bool]:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--headless" in args:
        return True
    if "--no-headless" in args:
        return False
    return None


def resolve_profile_name(
    explicit: Optional[str] = None,
    argv: Optional[list] = None,
) -> str:
    """
    Resolve canonical profile name.

    Priority:
      1. explicit argument
      2. HUNAV_ISAAC_PROFILE env
      3. CLI (--profile / --debug / --laptop)
      4. default (lab)
    """
    if explicit is not None and str(explicit).strip() != "":
        return normalize_profile_name(explicit)

    env_profile = os.environ.get("HUNAV_ISAAC_PROFILE")
    if env_profile is not None and env_profile.strip() != "":
        return normalize_profile_name(env_profile)

    argv_profile = _profile_from_argv(argv)
    if argv_profile is not None:
        return normalize_profile_name(argv_profile)

    return DEFAULT_PROFILE


def apply_profile_to_environ(
    profile: Optional[str] = None,
    headless: Optional[bool] = None,
) -> str:
    """
    Write resolved profile (and optional headless override) into the environment
    so a later import of teleop_hunav_sim sees them before SimulationApp starts.
    Returns the canonical profile name.
    """
    name = resolve_profile_name(explicit=profile)
    os.environ["HUNAV_ISAAC_PROFILE"] = name
    if headless is True:
        os.environ["HUNAV_ISAAC_HEADLESS"] = "1"
    elif headless is False:
        os.environ["HUNAV_ISAAC_HEADLESS"] = "0"
    return name


def build_simulation_config(
    profile: Optional[str] = None,
    headless: Optional[bool] = None,
    argv: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build SimulationApp CONFIG dict for the selected profile.

    headless override priority (first set wins):
      1. explicit headless= argument
      2. HUNAV_ISAAC_HEADLESS env
      3. CLI --headless / --no-headless
      4. profile default
    """
    name = resolve_profile_name(explicit=profile, argv=argv)
    config = copy.deepcopy(PROFILES[name])

    override = headless
    if override is None:
        override = _truthy_env(os.environ.get("HUNAV_ISAAC_HEADLESS"))
    if override is None:
        override = _headless_from_argv(argv)
    if override is not None:
        config["headless"] = bool(override)

    # Stash resolved name for logging (not passed to SimulationApp).
    config["_profile"] = name
    return config


def simulation_app_kwargs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Strip internal keys before passing to SimulationApp(...)."""
    if config is None:
        config = build_simulation_config()
    return {k: v for k, v in config.items() if not k.startswith("_")}


def describe_profiles() -> str:
    lines = [
        "SimulationApp profiles (HUNAV_ISAAC_PROFILE / --profile):",
        f"  default|lab   — { _LAB_CONFIG['width'] }x{ _LAB_CONFIG['height'] }, "
        f"headless={_LAB_CONFIG['headless']}, renderer={_LAB_CONFIG['renderer']}",
        f"  debug|laptop  — { _DEBUG_CONFIG['width'] }x{ _DEBUG_CONFIG['height'] }, "
        f"headless={_DEBUG_CONFIG['headless']}, renderer={_DEBUG_CONFIG['renderer']}",
        "Overrides: HUNAV_ISAAC_HEADLESS=0|1, --headless, --no-headless, --debug",
    ]
    return "\n".join(lines)


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared profile CLI flags to an ArgumentParser."""
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_ALIASES.keys()),
        default=None,
        help=(
            "SimulationApp profile: default|lab (1280x720 windowed RaytracedLighting) "
            "or debug|laptop (960x540 headless, lighter for 8GB VRAM). "
            "Also via HUNAV_ISAAC_PROFILE."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shorthand for --profile debug (laptop-friendly SimulationApp settings)",
    )
    parser.add_argument(
        "--laptop",
        action="store_true",
        help="Shorthand for --profile laptop (alias of debug)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force SimulationApp headless=True (also HUNAV_ISAAC_HEADLESS=1)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Force SimulationApp headless=False (also HUNAV_ISAAC_HEADLESS=0)",
    )


if __name__ == "__main__":
    # Smoke: python3 -m hunav_isaac_wrapper.sim_app_config [--profile debug]
    p = argparse.ArgumentParser(description="Print resolved SimulationApp CONFIG")
    add_profile_arguments(p)
    args = p.parse_args()
    explicit = args.profile
    if args.debug or args.laptop:
        explicit = DEBUG_PROFILE if args.debug else "laptop"
    headless = True if args.headless else (False if args.no_headless else None)
    cfg = build_simulation_config(profile=explicit, headless=headless)
    import json

    print(json.dumps(cfg, indent=2))
    print("--- kwargs ---")
    print(json.dumps(simulation_app_kwargs(cfg), indent=2))

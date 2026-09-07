#!/usr/bin/env python3
"""
world_builder.py

A simple module that loads a USD stage (map) from disk, replacing the current stage.
"""

import os
import omni
from pxr import Sdf, Usd

# PATCH (isaac-social-nav): hospital*.usd still ships Isaac 4.2 Nucleus material
# URLs (omniverse://localhost/.../Isaac/4.2/...). Those 404 on Isaac 6.0 and paint
# the whole scene red. Same assets live under get_assets_root_path() (CDN Isaac/6.0).
_LEGACY_ISAAC_PREFIXES = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.2",
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.5",
    "omniverse://localhost/NVIDIA/Assets/Isaac/5.0",
    "omniverse://localhost/NVIDIA/Assets/Isaac/5.1",
)

# CUCR ports that compose /World/DomeLight + /World/DistantLight in the stage USD.
# Default viewport rig on top of those lights blows the scene out (bookstore #58
# milky wash). Hospital/museum keep the Default rig.
_STAGE_LIGHT_WORLDS = frozenset(
    {
        "bookstore",
        "house_museum",
        "small_house",
        "small_warehouse",
    }
)


def _remap_legacy_isaac_asset_paths(stage: Usd.Stage) -> int:
    """Rewrite dead Nucleus Isaac 4.x/5.x asset URLs → current CDN root. Returns count."""
    try:
        from isaacsim.storage.native import get_assets_root_path
    except Exception as exc:
        print(f"[WorldBuilder] asset remap skipped (no assets root): {exc}")
        return 0

    new_root = (get_assets_root_path() or "").rstrip("/")
    if not new_root:
        print("[WorldBuilder] asset remap skipped: empty assets root")
        return 0

    changed = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != "asset":
                continue
            val = attr.Get()
            if val is None:
                continue
            old = str(getattr(val, "path", val) or "")
            if not old:
                continue
            for prefix in _LEGACY_ISAAC_PREFIXES:
                if old.startswith(prefix):
                    new = new_root + old[len(prefix) :]
                    attr.Set(Sdf.AssetPath(new))
                    changed += 1
                    break
    if changed:
        print(
            f"[WorldBuilder] remapped {changed} legacy Isaac asset path(s) → {new_root}"
        )
    return changed


def apply_viewport_defaults(map_name: str | None = None) -> None:
    """
    PATCH (isaac-social-nav): force free Perspective camera + sane viewport lighting.

    Hospital USD activates Camera_01 (locked) and Stage Lights — use Default rig.
    CUCR bookstore/residential stages ship DomeLight in the USD; Default rig
    on top washes them out. Office and hospital/museum use the Default rig.
    """
    use_stage = map_name in _STAGE_LIGHT_WORLDS
    try:
        import omni.kit.actions.core

        reg = omni.kit.actions.core.get_action_registry()
        if use_stage:
            action = reg.get_action(
                "omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage"
            )
            if action is not None:
                action.execute()
                print(f"[WorldBuilder] viewport lighting → Stage ({map_name})")
        else:
            action = reg.get_action(
                "omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig"
            )
            if action is not None:
                action.execute("Default")
                print("[WorldBuilder] viewport lighting → Default")
    except Exception as exc:
        print(f"[WorldBuilder] lighting default failed: {exc}")

    # Camera: free Perspective (not hospital Camera_01)
    try:
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        if vp is not None:
            if hasattr(vp, "set_active_camera"):
                vp.set_active_camera("/OmniverseKit_Persp")
            else:
                vp.camera_path = "/OmniverseKit_Persp"
            print("[WorldBuilder] viewport camera → /OmniverseKit_Persp")
    except Exception as exc:
        print(f"[WorldBuilder] perspective camera failed: {exc}")


class WorldBuilder:
    """
    Loads an entire USD stage from disk.
    
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.usd_context = omni.usd.get_context()

    def load_map(self, map_name: str):
        """
        Looks for `map_name.usd` inside the 'worlds' folder under base_path and opens it.
        """
        from .resource_paths import resolve_world_usd

        map_path = resolve_world_usd(
            map_name, os.path.join(self.base_path, "worlds")
        )
        if os.path.exists(map_path):
            self.usd_context.open_stage(map_path)
            print(f"Map '{map_name}' loaded from: {map_path}")
            stage = self.usd_context.get_stage()
            if stage is not None:
                _remap_legacy_isaac_asset_paths(stage)
            apply_viewport_defaults(map_name)
        else:
            print(f"[WorldBuilder] Error: map '{map_name}' not found at {map_path}")

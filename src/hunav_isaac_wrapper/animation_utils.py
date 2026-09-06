#!/usr/bin/env python3
"""
animation_utils.py

This module contains helper functions for handling animations and animation graphs,
including functions to create/apply animations, set up retargeting, and to find
Skeleton and SkelRoot prims.
"""

import os
import tempfile

import omni.kit.commands
from pxr import Sdf, Usd, UsdSkel


def find_skeleton_path(agentPrim):
    """
    Recursively searches for a prim of type "Skeleton" within agentPrim.

    """
    if agentPrim.GetTypeName() == "Skeleton":
        return agentPrim.GetPath()
    for child in agentPrim.GetChildren():
        if not child.IsValid():
            continue
        child_path_str = child.GetPath().pathString
        if ("Looks" in child_path_str) or ("CharacterAnimation" in child_path_str):
            continue
        if child.GetTypeName() == "Skeleton":
            return child.GetPath()
        result = find_skeleton_path(child)
        if result:
            return result
    print(
        f"Warning: No Skeleton found for {agentPrim.GetPath()}, using /Root by default."
    )
    return Sdf.Path(f"{agentPrim.GetPath()}/Root")


def find_skelroot_path(agentPrim):
    """
    Recursively searches for a SkelRoot prim within the children of agentPrim.

    """
    for child in agentPrim.GetChildren():
        if not child.IsValid():
            continue
        child_path_str = child.GetPath().pathString

        if ("Looks" in child_path_str) or ("CharacterAnimation" in child_path_str):
            continue

        if "ManRoot" in child_path_str:
            for grandchild in child.GetChildren():
                if grandchild.IsValid() and grandchild.GetTypeName() == "SkelRoot":
                    return grandchild.GetPath()

        if child.GetTypeName() == "SkelRoot":
            return child.GetPath()

        result = find_skelroot_path(child)
        if result:
            return result

    print(
        f"Warning: No SkelRoot found within {agentPrim.GetPath()}, using fallback /Root."
    )
    return Sdf.Path(f"{agentPrim.GetPath()}/Root")


def create_animation(stage, animation_path, source_path):
    """
    Creates an animation prim (of type "SkelAnimation") at the specified path,
    referencing the source animation USD file.
    """
    animation = stage.DefinePrim(animation_path, "SkelAnimation")
    animation.GetReferences().AddReference(source_path)
    return animation


def create_agent_animation_graph(stage, agent_prim, idle_anim_path, walk_anim_path):
    """
    Creates an AnimationGraph for the given agent prim that blends between idle
    and walk animations.

    Returns:
        The Sdf.Path to the newly created AnimationGraph prim.
    """
    # Define a unique path for the AnimationGraph prim under the agent's prim
    graph_path = Sdf.Path(f"{str(agent_prim.GetPath())}/AnimationGraph")
    anim_graph_prim = stage.DefinePrim(graph_path, "AnimationGraph")

    # ---------------------------------------------------------------------------
    # ORIGINALLY (upstream v2.0): graph was created, then Blend / clips / ReadVariable
    # for "speed" — but anim:graph:variable:speed was NEVER declared on the graph
    # prim at authoring time (only set later, optionally, in set_anim_graph_speed).
    # Had to be patched because Isaac 6.0 failed to compile the graph:
    #   Failed to compile asset '.../AnimationGraph'
    #   Type mismatch for variable speed in Character::SetVariable
    # ---------------------------------------------------------------------------
    # PATCH (isaac-social-nav): declare custom *uniform* float
    # anim:graph:variable:speed before ReadVariable (matches Isaac AnimGraph test
    # USDs). Important so walk/idle blend compiles and set_variable("speed", ...) works.
    # ---------------------------------------------------------------------------
    speed_var = anim_graph_prim.CreateAttribute(
        "anim:graph:variable:speed",
        Sdf.ValueTypeNames.Float,
        custom=True,
        variability=Sdf.VariabilityUniform,
    )
    speed_var.Set(0.0)

    # Create and configure child nodes:
    # Blend node
    blend_path = graph_path.AppendChild("Blend")
    blend_prim = stage.DefinePrim(blend_path, "Blend")
    # Idle AnimationClip node
    idle_clip_path = graph_path.AppendChild("IdleLoop")
    idle_clip_prim = stage.DefinePrim(idle_clip_path, "AnimationClip")
    idle_clip_prim.CreateRelationship("inputs:animationSource").SetTargets(
        [idle_anim_path]
    )
    # Walk AnimationClip node
    walk_clip_path = graph_path.AppendChild("WalkLoop")
    walk_clip_prim = stage.DefinePrim(walk_clip_path, "AnimationClip")
    walk_clip_prim.CreateRelationship("inputs:animationSource").SetTargets(
        [walk_anim_path]
    )
    # ReadVariable node for speed
    speed_node_path = graph_path.AppendChild("speed")
    speed_node_prim = stage.DefinePrim(speed_node_path, "ReadVariable")
    # ORIGINALLY: CreateAttribute("inputs:variableName", Token).Set("speed")
    # PATCH: also mark custom + VariabilityUniform for Isaac 6.0 AnimGraph consistency.
    speed_node_prim.CreateAttribute(
        "inputs:variableName",
        Sdf.ValueTypeNames.Token,
        custom=True,
        variability=Sdf.VariabilityUniform,
    ).Set("speed")

    # Set up connections:
    blend_prim.CreateRelationship("inputs:blendWeight").SetTargets([speed_node_path])
    blend_prim.CreateRelationship("inputs:pose0").SetTargets([idle_clip_path])
    blend_prim.CreateRelationship("inputs:pose1").SetTargets([walk_clip_path])
    anim_graph_prim.CreateRelationship("inputs:pose").SetTargets([blend_path])

    # Bind the AnimationGraph to the agent's skeleton
    skel_path = find_skeleton_path(agent_prim)
    anim_graph_prim.CreateRelationship("skel:skeleton").SetTargets([skel_path])

    print(f"Created AnimationGraph for {agent_prim.GetPath()} at {graph_path}")
    return graph_path


def apply_animation_graph(agent_prim, graph_path):
    """
    Applies an AnimationGraph to the given agent's prim by executing the built-in command.
    """
    omni.kit.commands.execute(
        "ApplyAnimationGraphAPICommand",
        paths=[Sdf.Path(agent_prim.GetPath())],
        animation_graph_path=graph_path,
    )
    print(f"Applied AnimationGraph ({graph_path}) to {agent_prim.GetPath()}")


def _export_skelanim_usd(stage, src_prim_path: str, out_path: str) -> bool:
    """
    Write an authored SkelAnimation prim to a standalone USD file.

    Copies joints plus rotations/translations/scales time samples (and a default
    value at the first sample) so AnimationClip can load the clip via reference.
    """
    src = stage.GetPrimAtPath(src_prim_path)
    if not src or not src.IsValid():
        print(f"export_skelanim: missing prim {src_prim_path}")
        return False
    anim = UsdSkel.Animation(src)
    if not anim:
        print(f"export_skelanim: not a SkelAnimation: {src_prim_path}")
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    exp = Usd.Stage.CreateNew(out_path)
    dst = UsdSkel.Animation.Define(exp, "/Root")
    joints = anim.GetJointsAttr().Get()
    if joints:
        dst.GetJointsAttr().Set(joints)

    for name in ("rotations", "translations", "scales"):
        src_attr = src.GetAttribute(name)
        if not src_attr or not src_attr.IsValid():
            continue
        samples = src_attr.GetTimeSamples()
        if not samples:
            continue
        dst_attr = exp.GetPrimAtPath("/Root").GetAttribute(name)
        first = src_attr.Get(samples[0])
        if first is not None:
            dst_attr.Set(first)
        for t in samples:
            dst_attr.Set(src_attr.Get(t), t)

    exp.SetDefaultPrim(exp.GetPrimAtPath("/Root"))
    exp.GetRootLayer().Save()
    return True


def materialize_retargeted_animation_references(
    stage, inline_anim_paths, export_dir
):
    """
    Replace inline retargeted SkelAnimation prims with file-referenced copies.

    ---------------------------------------------------------------------------
    PATCH (isaac-social-nav): Isaac 6.0 AnimGraph AnimationClip plays referenced
    SkelAnimation USDs, but clips authored inline by CreateRetargetAnimationsCommand
    stay at bind pose (T-pose) even when the USD time samples contain motion.
    Export + AddReference matches NVIDIA play_animation.usda / source People clips.
    ---------------------------------------------------------------------------
    """
    os.makedirs(export_dir, exist_ok=True)
    result = {}
    for key, inline_path in inline_anim_paths.items():
        inline_path = str(inline_path)
        leaf = inline_path.rstrip("/").split("/")[-1]
        out_file = os.path.join(export_dir, f"{leaf}.skelanim.usd")
        if not _export_skelanim_usd(stage, inline_path, out_file):
            result[key] = inline_path
            continue
        # Swap the inline prim for a reference to the exported file.
        if stage.GetPrimAtPath(inline_path):
            stage.RemovePrim(inline_path)
        create_animation(stage, inline_path, out_file)
        result[key] = inline_path
        print(f"Materialized retargeted clip {inline_path} -> {out_file}")
    return result


def setup_anim_retargeting(
    stage, agent_prim, source_animation_dict, target_animation_parent_path
):
    """
    Sets up retargeting for the target agent's animation by using the default source biped.

    Args:
        stage: The USD stage.
        agent_prim: The target agent prim for which retargeting will be applied.
        source_animation_dict: A dictionary with source animation paths.
        target_animation_parent_path: The USD path under which retargeted animations will be created.

    Returns:
        Dict mapping 0/1 -> idle/walk prim paths (file-referenced after materialize),
        or None on failure.
    """
    source_agent_prim = stage.GetPrimAtPath("/World/Biped_Setup/biped_demo_meters")
    if not source_agent_prim or not source_agent_prim.IsValid():
        print("Default biped prim not found.")
        return None

    source_skel_path = find_skeleton_path(source_agent_prim)
    if not source_skel_path:
        print("Default biped skeleton not found.")
        return None
    source_skel_str = str(source_skel_path)

    target_skel_path = find_skeleton_path(agent_prim)
    if not target_skel_path:
        print("Target skeleton not found.")
        return None
    target_skel_str = str(target_skel_path)

    source_anim_path = str(source_animation_dict[1])

    omni.kit.commands.execute(
        "CreateRetargetAnimationsCommand",
        source_skeleton_path=source_skel_str,
        target_skeleton_path=target_skel_str,
        source_animation_paths=[source_anim_path],
        target_animation_parent_path=target_animation_parent_path,
        set_root_identity=False,
    )
    omni.kit.commands.execute(
        "CreateRetargetAnimationsCommand",
        source_skeleton_path=source_skel_str,
        target_skeleton_path=target_skel_str,
        source_animation_paths=[str(source_animation_dict[0])],
        target_animation_parent_path=target_animation_parent_path,
        set_root_identity=False,
    )

    # ORIGINALLY: left inline SkelAnimation prims under target_animation_parent_path
    # and pointed AnimationClip at them. On Isaac 6.0 those inline clips do not
    # drive AnimationGraph (bind pose / T-pose) even with valid time samples.
    # PATCH: export each clip to a temp USD and re-reference it in-place.
    inline = {
        0: f"{target_animation_parent_path}/IdleLoop",
        1: f"{target_animation_parent_path}/WalkLoop",
    }
    agent_leaf = str(agent_prim.GetPath()).strip("/").replace("/", "_")
    export_dir = os.path.join(
        tempfile.gettempdir(), "hunav_isaac_retarget", agent_leaf
    )
    return materialize_retargeted_animation_references(stage, inline, export_dir)


def set_anim_graph_speed(stage, anim_graph_character, graph_path, speed_value):
    """
    Sets the 'anim:graph:variable:speed' attribute on the AnimationGraph prim.
    If the attribute does not exist, it is created as a uniform float (Isaac 6.0).

    Args:
        stage: The USD stage.
        anim_graph_character: The anim graph character instance.
        graph_path: The Sdf.Path to the AnimationGraph prim.
        speed_value: The float value to set for the 'speed' variable.
    """
    graph_prim = stage.GetPrimAtPath(graph_path)
    if not graph_prim or not graph_prim.IsValid():
        print(f"set_anim_graph_speed: AnimationGraph prim not found at {graph_path}")
        return

    speed_attr = graph_prim.GetAttribute("anim:graph:variable:speed")
    if not speed_attr or not speed_attr.IsValid():
        # ORIGINALLY (upstream v2.0):
        # speed_attr = graph_prim.CreateAttribute(
        #     "anim:graph:variable:speed", Sdf.ValueTypeNames.Float, custom=True
        # )
        # Had to be patched: Isaac 6.0 expects *uniform* float graph variables.
        speed_attr = graph_prim.CreateAttribute(
            "anim:graph:variable:speed",
            Sdf.ValueTypeNames.Float,
            custom=True,
            variability=Sdf.VariabilityUniform,
        )
    # ---------------------------------------------------------------------------
    # ORIGINALLY (upstream v2.0):
    #   anim_graph_character.set_variable("speed", speed_value)
    # Had to be patched because callers pass numpy.float64 from np.clip(...);
    # Character::SetVariable expects C++ float → "Type mismatch for variable speed".
    # ---------------------------------------------------------------------------
    # PATCH (isaac-social-nav): cast to Python float, sync USD attr, then set_variable.
    # Important so walk/idle blend weight updates without compile/type errors.
    # ---------------------------------------------------------------------------
    speed_f = float(speed_value)
    if speed_attr and speed_attr.IsValid():
        speed_attr.Set(speed_f)
    anim_graph_character.set_variable("speed", speed_f)

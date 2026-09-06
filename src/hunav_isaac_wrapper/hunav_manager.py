#!/usr/bin/env python3
"""
hunav_manager.py

Contains the HuNavManager class for spawning and managing Hunav agents,
setting up animations (including retargeting), and handling physics.
"""

import os
import math
import random
import time
import yaml
import numpy as np
import subprocess, signal
from typing import Tuple, List, Optional

# ROS messages
import rclpy
from geometry_msgs.msg import Quaternion, Pose, Point
from hunav_msgs.srv import ComputeAgents
from hunav_msgs.msg import Agent, Agents, AgentBehavior
from std_msgs.msg import Header

# Isaac Sim imports
import omni
import omni.kit.commands
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension

from pxr import Sdf, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import carb

# Import auxiliary animation functions
from .animation_utils import *
from .occupancy_path import OccupancyMap, resolve_map_yaml
from .behavior_labels import BehaviorLabelOverlay, behavior_labels_enabled

# ---------------------------------------------------------------------------
# ORIGINALLY (upstream v2.0): only retarget was enabled here, then graph imported:
#   enable_extension("omni.anim.retarget.core")
#   import omni.anim.graph.core as ag
# Had to be patched because on Isaac 6.0.1 (isaacsim.exp.base.python) 
# omni.anim.graph.core is NOT loaded by default → ModuleNotFoundError.
# Prefer Kit-start --enable via sim_app_config extra_args; also request both
# extensions here. Do NOT pump app.update() after enable_extension — that
# crashed Kit (OmniGraph / exit 139) during E2E on this host.
# ---------------------------------------------------------------------------
# PATCH (isaac-social-nav): enable graph + retarget before importing ag so
# AnimationGraph / retarget APIs resolve on Isaac 6.0.
# ---------------------------------------------------------------------------
enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.retarget.core")

import omni.anim.graph.core as ag


class HuNavManager:
    """
    Manages HunavSim agents by reading configuration from a YAML file,
    spawning agents as SkelRoot prims, setting up animations (and retargeting),
    and handling ROS 2 communications.
    """

    def __init__(self, node, world, config_file_path, robot_prim_path, robot):
        self.node = node
        self.stage = world.stage
        self.robot_prim_path = robot_prim_path
        self.robot_obj = robot
        self.world = world
        self.config_file_path = config_file_path

        self.assets_root = get_assets_root_path()
        self._usd_context = omni.usd.get_context()

        # HuNavSim's ROS 2 service client
        self.compute_agents_client = self.node.create_client(
            ComputeAgents, "/compute_agents"
        )

        # List of target model assets
        character_root_path = os.path.join(self.assets_root, "Isaac/People/Characters/")

        character_models = [
            "F_Business_02/F_Business_02.usd",
            "F_Medical_01/F_Medical_01.usd",
            "M_Medical_01/M_Medical_01.usd",
            "male_adult_construction_01_new/male_adult_construction_01_new.usd",
            "male_adult_construction_05_new/male_adult_construction_05_new.usd",
            "female_adult_police_01_new/female_adult_police_01_new.usd",
            "female_adult_police_02/female_adult_police_02.usd",
            "female_adult_police_03_new/female_adult_police_03_new.usd",
            "male_adult_police_04/male_adult_police_04.usd",
            "original_female_adult_business_02/female_adult_business_02.usd",
            "original_female_adult_medical_01/female_adult_medical_01.usd",
        ]

        self.target_model_paths = [
            f"{character_root_path}{model}" for model in character_models
        ]
        
        # Mapping from skin ID to character model index
        # This allows agents to specify which character model to use via the 'skin' field
        # in their configuration. Valid options:
        #   0-10: Specific character models (see mapping below)
        #   "random": Random character model selection
        self.skin_to_model_mapping = {
            1: 0,   # F_Business_02
            2: 1,   # F_Medical_01
            3: 2,   # M_Medical_01
            4: 3,   # male_adult_construction_01_new
            5: 4,   # male_adult_construction_05_new
            6: 5,   # female_adult_police_01_new
            7: 6,   # female_adult_police_02
            8: 7,   # female_adult_police_03_new
            9: 8,   # male_adult_police_04
            10: 9,  # original_female_adult_business_02
            11: 10, # original_female_adult_medical_01
        }

        # Data holders
        self.agents = []
        self.agent_initial_states = []
        self.animationDict = {}
        self._hunav_processes = []
        self.bound_animations = {}
        self.flag_anim = {}
        
        # Orientation smoothing
        self.agent_previous_orientations = {}  # Store previous orientations for smoothing
        self.orientation_smoothing_factor = 0.15  # Lower = smoother but more lag (0.05-0.3 range)
        # PATCH (isaac-social-nav): near-robot diagnostics / prior XY for logging.
        self.agent_previous_positions = {}
        self.agent_previous_deltas = {}
        self._reaction_log_prev = {}
        self._reaction_log_warned = False

        self.robot_prim = None
        # PATCH (isaac-social-nav): occupancy A* goal expansion cache (per agent name).
        self._occupancy_map = None
        self._planned_goals_cache = {}
        # Viewport behavior name overlays (GUI demos; off when headless).
        self._behavior_labels = BehaviorLabelOverlay()
        self._behavior_labels.set_enabled(behavior_labels_enabled())

        if config_file_path is not None:
            self.config = self._load_yaml(config_file_path)
            self._maybe_load_occupancy_map()
        else:
            self.config = None

        # ---------------------------------------------------------------------------
        # ORIGINALLY (upstream v2.0) — Isaac 4.5/5.x People content path:
        # self.default_biped_usd = os.path.join(
        #     self.assets_root, "Isaac/People/Characters/Biped_Setup.usd"
        # )
        # Had to be patched because Isaac 6.0 CDN returns HTTP 404 for that path
        # (People/Characters/Biped_Setup.usd removed from the 6.0 tree).
        # ---------------------------------------------------------------------------
        # First PATCH tried AnimGraph/105.0/Test/Graph/Isaac/Biped_Setup.usd (HTTP 200),
        # but that skeleton only shares 1 retarget tag ("Head") with Isaac 6 People
        # RL_BoneRoot skins → CreateRetargetAnimationsCommand yields near-static clips
        # and agents stay in T-pose.
        # ---------------------------------------------------------------------------
        # PATCH (isaac-social-nav): use Isaac 5.1 People Biped_Setup (still HTTP 200).
        # Shares ~51 retarget tags with Isaac 6.0 People characters so walk/idle
        # retarget actually carries motion. Pair with materialized file-referenced
        # clips in animation_utils.setup_anim_retargeting (inline clips won't play).
        # ---------------------------------------------------------------------------
        self.default_biped_usd = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
            "Assets/Isaac/5.1/Isaac/People/Characters/Biped_Setup.usd"
        )

    def _load_yaml(self, relative_path):
        full_path = os.path.join(os.path.dirname(__file__), relative_path)
        with open(full_path, "r") as file:
            return yaml.safe_load(file)

    def _iter_maps_dirs(self):
        """Source src/maps first (install share can lag), then share/.../maps."""
        src_maps = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "maps")
        )
        if os.path.isdir(src_maps):
            yield src_maps
        try:
            result = subprocess.run(
                ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
                capture_output=True,
                text=True,
                check=True,
            )
            share_maps = os.path.join(
                result.stdout.strip(), "share", "hunav_isaac_wrapper", "maps"
            )
            if os.path.isdir(share_maps) and os.path.abspath(share_maps) != src_maps:
                yield share_maps
        except (subprocess.CalledProcessError, FileNotFoundError):
            return

    def _find_maps_dir(self) -> str:
        """Locate share/.../maps or source src/maps without importing teleop (circular)."""
        for maps_dir in self._iter_maps_dirs():
            return maps_dir
        raise FileNotFoundError("Could not locate hunav_isaac_wrapper maps/ directory")

    def _maybe_load_occupancy_map(self) -> None:
        """
        PATCH (isaac-social-nav): optional A* on ROS occupancy maps.

        When ``plan_goals_on_map`` is true, sparse YAML goals are expanded into
        corridor-following waypoint chains so SFM no longer takes wall-cutting chords.
        """
        self._occupancy_map = None
        self._planned_goals_cache = {}
        if self.config is None:
            return
        params = self.config.get("hunav_loader", {}).get("ros__parameters", {})
        if not params.get("plan_goals_on_map", False):
            return
        map_key = params.get("map_yaml") or params.get("map") or "museum"
        inflation = float(params.get("plan_inflation_radius_m", 0.35))
        last_exc = None
        for maps_dir in self._iter_maps_dirs():
            try:
                yaml_path = resolve_map_yaml(str(map_key), maps_dir)
                self._occupancy_map = OccupancyMap.from_yaml(
                    yaml_path, inflation_radius_m=inflation
                )
                print(
                    f"[HuNavManager] Occupancy planner loaded: {yaml_path} "
                    f"(inflation={inflation} m)"
                )
                return
            except FileNotFoundError as exc:
                last_exc = exc
            except Exception as exc:
                last_exc = exc
                break
        print(
            f"[HuNavManager] WARNING: plan_goals_on_map enabled but map load failed: "
            f"{last_exc}"
        )
        self._occupancy_map = None

    def slerp_quaternions(self, q1: Gf.Quatf, q2: Gf.Quatf, t: float) -> Gf.Quatf:
        """
        Spherical linear interpolation between two quaternions for smooth rotation.
        
        Args:
            q1: Starting quaternion
            q2: Target quaternion  
            t: Interpolation factor (0.0 = q1, 1.0 = q2)
            
        Returns:
            Interpolated quaternion
        """
        # Ensure we take the shortest path by checking dot product
        dot = q1.GetReal() * q2.GetReal() + sum(a * b for a, b in zip(q1.GetImaginary(), q2.GetImaginary()))
        
        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = Gf.Quatf(-q2.GetReal(), -q2.GetImaginary()[0], -q2.GetImaginary()[1], -q2.GetImaginary()[2])
            dot = -dot
        
        # If quaternions are very close, use linear interpolation to avoid division by zero
        if dot > 0.9995:
            result_real = q1.GetReal() + t * (q2.GetReal() - q1.GetReal())
            result_imag = [
                q1.GetImaginary()[i] + t * (q2.GetImaginary()[i] - q1.GetImaginary()[i])
                for i in range(3)
            ]
            result = Gf.Quatf(result_real, result_imag[0], result_imag[1], result_imag[2])
            return result.GetNormalized()
        
        # Calculate spherical interpolation
        theta_0 = math.acos(abs(dot))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result_real = s0 * q1.GetReal() + s1 * q2.GetReal()
        result_imag = [
            s0 * q1.GetImaginary()[i] + s1 * q2.GetImaginary()[i]
            for i in range(3)
        ]
        
        return Gf.Quatf(result_real, result_imag[0], result_imag[1], result_imag[2]).GetNormalized()

    def normalize_angle(self, a: float) -> float:
        value = a
        while value <= -math.pi:
            value += 2 * math.pi
        while value > math.pi:
            value -= 2 * math.pi
        return value

    def initialize_hunav_nodes(self):
        process_1 = subprocess.Popen(
            [
                "ros2",
                "run",
                "hunav_agent_manager",
                "hunav_loader",
                "--ros-args",
                "--params-file",
                self.config_file_path,
            ],
            preexec_fn=os.setsid,
        )
        process_2 = subprocess.Popen(
            [
                "ros2",
                "run",
                "hunav_agent_manager",
                "hunav_agent_manager",
                "--ros-args",
                "--params-file",
                self.config_file_path,
            ],
            preexec_fn=os.setsid,
        )
        # PATCH (isaac-social-nav): do not auto-start hunav_evaluator under Isaac.
        # Isaac's bundled Python hits pandas/numpy ABI errors; run the evaluator
        # from a system ROS shell when collecting metrics (HUNAV_START_EVALUATOR=1
        # opts back into the old spawn for Gazebo-style workflows).
        self._hunav_processes = [process_1, process_2]
        if os.environ.get("HUNAV_START_EVALUATOR", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            process_3 = subprocess.Popen(
                ["ros2", "run", "hunav_evaluator", "hunav_evaluator_node"],
                preexec_fn=os.setsid,
            )
            self._hunav_processes.append(process_3)

    def close_hunav_nodes(self):
        for process in self._hunav_processes:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        self._hunav_processes = []

    def initialize_agents(self):
        """
        Reads the YAML configuration for agents, spawns them, and sets up their animations
        using a two-level hierarchy. For each agent, the outer container prim (type "Xform") is updated by
        compute_agents (global transform), while the inner child prim (type "SkelRoot") is
        animated via an AnimationGraph created beforehand.
        """
        if self.config is None:
            print("[HuNavManager] No config loaded, skipping agent creation.")
            return

        agent_configs = self.config["hunav_loader"]["ros__parameters"]["agents"]

        # Create animations using the auxiliary module functions
        animations_path = os.path.join(
            self.assets_root, "Isaac", "People", "Animations"
        )
        walk_anim = create_animation(
            self.stage,
            "/World/Animations/WalkLoop",
            os.path.join(animations_path, "stand_walk_loop_in_place.skelanim.usd"),
        )
        idle_anim = create_animation(
            self.stage,
            "/World/Animations/IdleLoop",
            os.path.join(animations_path, "stand_idle_loop.skelanim.usd"),
        )
        self.source_animation_dict = {0: idle_anim.GetPath(), 1: walk_anim.GetPath()}
        # Set up a rotation for upright orientation
        rotX = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat()
        rotXQ = Gf.Quatf(rotX)

        # Spawn the default biped for skeletal binding
        init_pos_src = Gf.Vec3d(0, 0, 0)
        init_rot_src = Gf.Quatf(1, 0, 0, 0) * rotXQ
        source_prim_path = "/World/Biped_Setup"
        source_agent_prim = self.stage.DefinePrim(source_prim_path, "SkelRoot")
        source_agent_prim.GetReferences().AddReference(self.default_biped_usd)
        xform_src = UsdGeom.Xformable(source_agent_prim)
        found_translate = False
        for op in xform_src.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                op.Set(init_pos_src)
                found_translate = True
                break
        if not found_translate:
            xform_src.AddTranslateOp().Set(init_pos_src)
        xform_src.AddOrientOp().Set(init_rot_src)
        source_agent_prim.GetAttribute("visibility").Set("invisible")
        source_agent_prim.CreateAttribute(
            "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool
        ).Set(True)
        source_agent_prim.CreateAttribute(
            "physxContact:collisionEnabled", Sdf.ValueTypeNames.Bool
        ).Set(False)

        asset_cycle = self.target_model_paths.copy()
        random.shuffle(asset_cycle)

        # For each agent defined in the agents_x.yaml:
        for agent_name in agent_configs:
            agent_cfg = self.config["hunav_loader"]["ros__parameters"][agent_name]

            # Use skin value to select specific character model
            skin_value = agent_cfg["skin"]
            asset_path = self.get_character_model_from_skin(skin_value)
            
            if asset_path is None:
                # Invalid skin value, fall back to round-robin selection
                self.node.get_logger().warn(
                    f"Invalid skin value '{skin_value}' for agent {agent_name}, "
                    f"falling back to random selection. Valid skin values are 0 (random) or 1-{len(self.target_model_paths)}"
                )
                if len(asset_cycle) == 0:
                    asset_cycle = self.target_model_paths.copy()
                    random.shuffle(asset_cycle)
                asset_path = asset_cycle.pop()
            else:
                if skin_value == 0:
                    self.node.get_logger().info(
                        f"Agent {agent_name} using random skin: {asset_path.split('/')[-2]}"
                    )
                else:
                    self.node.get_logger().info(
                        f"Agent {agent_name} using skin {skin_value}: {asset_path.split('/')[-2]}"
                    )
            
            init_pose = agent_cfg["init_pose"]
            # translation
            global_pos = Gf.Vec3d(init_pose["x"], init_pose["y"], init_pose["z"])

            h_rad = init_pose.get("h", 0.0)
            h_deg = h_rad * 180.0 / math.pi

            # X-tilt rotation (90°)
            rotX = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)
            qdX = rotX.GetQuat()  # this is a Gf.Quatd
            # convert to single‐precision quaternion:
            rotXQ = Gf.Quatf(qdX.GetReal(), Gf.Vec3f(qdX.GetImaginary()))

            # Z-heading rotation
            rotZ = Gf.Rotation(Gf.Vec3d(0, 0, 1), h_deg)
            qdZ = rotZ.GetQuat()
            rotZQ = Gf.Quatf(qdZ.GetReal(), Gf.Vec3f(qdZ.GetImaginary()))

            # combine: yaw then tilt
            global_rot = rotZQ * rotXQ

            # now apply to container:
            container_path = f"/World/Characters/{agent_name}"
            container = self.stage.DefinePrim(container_path, "Xform")
            xf = UsdGeom.Xformable(container)
            xf.AddTranslateOp().Set(global_pos)
            xf.AddOrientOp().Set(global_rot)
            
            # HuNav teleports agents each step — keep bodies kinematic / no gravity
            # so PhysX cannot sink them through the floor between updates.
            UsdPhysics.RigidBodyAPI.Apply(container).CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(container).CreateDisableGravityAttr(
                True
            )

            # Create the inner animated SkelRoot as a child of the container
            anim_path = container_path + "/Animation"
            agent_skelroot = self.stage.DefinePrim(anim_path, "SkelRoot")
            agent_skelroot.GetReferences().AddReference(asset_path)
            # Character meshes ship with colliders; PhysX obstacle rays would hit
            # the agent itself → SFM thrashing / spin-in-place against "walls".
            self._disable_collisions_recursive(container)

            # Retarget clips per character, then materialize as file references
            # (see setup_anim_retargeting PATCH). Sharing one character's clips
            # across different RL skins can leave others in bind pose.
            target_animation_parent_path = (
                f"{container_path}/RetargetedAnimations"
            )
            self.stage.DefinePrim(target_animation_parent_path, "Xform")
            animation_dict = setup_anim_retargeting(
                self.stage,
                agent_skelroot,
                self.source_animation_dict,
                target_animation_parent_path,
            )
            if not animation_dict:
                animation_dict = {
                    0: f"{target_animation_parent_path}/IdleLoop",
                    1: f"{target_animation_parent_path}/WalkLoop",
                }

            # Create and apply AnimationGraph
            anim_graph_path = create_agent_animation_graph(
                self.stage,
                agent_skelroot,
                idle_anim_path=animation_dict[0],
                walk_anim_path=animation_dict[1],
            )
            apply_animation_graph(
                self.stage.GetPrimAtPath(find_skelroot_path(agent_skelroot)),
                anim_graph_path,
            )
            self.bound_animations[agent_skelroot.GetPath()] = anim_graph_path

            self.flag_anim[agent_skelroot.GetPath()] = False

            # Add the container (global transform) to the agents list
            self.agents.append(container)
            self.agent_initial_states.append(
                {"position": global_pos, "orientation": global_rot}
            )

        # Set up the robot prim
        if self.robot_prim_path:
            rp = self.stage.GetPrimAtPath(self.robot_prim_path)
            if rp.IsValid():
                self.robot_prim = rp
            else:
                print(
                    f"[HuNavManager] Warning: no valid robot prim at {self.robot_prim_path}"
                )
        else:
            print("[HuNavManager] no robot_prim_path provided")

        # Viewport labels: A{id} · BEHAVIOR (DrawLabel above each agent).
        if behavior_labels_enabled():
            self._behavior_labels.set_enabled(True)
            agent_ids = []
            params = self.config["hunav_loader"]["ros__parameters"]
            for name in params.get("agents") or []:
                cfg = params.get(name) or {}
                agent_ids.append(int(cfg.get("id", len(agent_ids) + 1)))
            if not agent_ids:
                agent_ids = list(range(1, len(self.agents) + 1))
            if self._behavior_labels.ensure_labels(agent_ids):
                for name in params.get("agents") or []:
                    cfg = params.get(name) or {}
                    aid = int(cfg.get("id", 0))
                    init = cfg.get("init_pose") or {}
                    beh = cfg.get("behavior") or {}
                    self._behavior_labels.update_label(
                        aid,
                        float(init.get("x", 0.0)),
                        float(init.get("y", 0.0)),
                        float(init.get("z", 0.0)),
                        self._parse_behavior_type(beh.get("type", 1)),
                        0,
                    )

    def reset_agent_states(self):
        for agent, init_state in zip(self.agents, self.agent_initial_states):
            agent.GetAttribute("xformOp:translate").Set(init_state["position"])
            agent.GetAttribute("xformOp:orient").Set(init_state["orientation"])
        
        self.agent_previous_orientations.clear()
        self.agent_previous_positions.clear()
        self.agent_previous_deltas.clear()
        print("[HuNavManager] agent states reset.")

    def clear_simulation(self):
        self.close_hunav_nodes()
        stage = self._usd_context.get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        for prim in world_prim.GetChildren():
            stage.RemovePrim(prim.GetPath())
        self.agents.clear()
        self.robot = None
        self.agent_initial_states.clear()
        self.animationDict.clear()
        self.agent_previous_orientations.clear()
        self.agent_previous_positions.clear()
        self.agent_previous_deltas.clear()
        try:
            self._behavior_labels.destroy()
        except Exception:
            pass

    def _disable_collisions_recursive(self, root_prim) -> None:
        """Turn off collision on an agent prim tree (HuNav moves agents kinematically)."""
        for prim in Usd.PrimRange(root_prim):
            try:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            except Exception:
                pass
            for attr_name in (
                "physics:collisionEnabled",
                "physxContact:collisionEnabled",
            ):
                try:
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        attr.Set(False)
                    else:
                        prim.CreateAttribute(
                            attr_name, Sdf.ValueTypeNames.Bool
                        ).Set(False)
                except Exception:
                    pass

    # Obstacle detection functions
    def generate_lasers(self, num_lasers: int) -> List[Gf.Vec3f]:
        """
        Generate ray directions evenly distributed over 360° in the global frame.
        """
        directions = []
        angle_increment = 360.0 / num_lasers
        for i in range(num_lasers):
            angle_rad = math.radians(i * angle_increment)
            x = math.cos(angle_rad)
            y = math.sin(angle_rad)
            directions.append(Gf.Vec3f(x, y, 0.0))
        return directions

    def get_closest_obstacles(
        self,
        agent_position: Gf.Vec3d,
        max_distance: float,
        sensor_offsets: List[float],
        num_lasers: int = 90,
    ) -> List[Tuple[float, Optional[Gf.Vec3f]]]:
        """
        Cast rays from the agent's sensor origins at multiple heights in fixed directions.
        For each ray direction, iterate over the provided sensor_offsets and select the hit
        with the smallest distance (if any).
        """
        directions = self.generate_lasers(num_lasers)
        scene_query = omni.physx.get_physx_scene_query_interface()
        closest_hits = []

        # Iterate over each ray direction
        for direction in directions:
            best_distance = max_distance
            best_hit = None
            hit_found = False
            # Test each sensor offset
            for offset in sensor_offsets:
                sensor_origin = Gf.Vec3f(
                    float(agent_position[0]),
                    float(agent_position[1]),
                    float(agent_position[2]) + offset,
                )
                hit = scene_query.raycast_closest(
                    sensor_origin, direction, max_distance
                )
                if hit.get("hit", False):
                    # Ignore self / other HuNav agents and near-contact noise.
                    coll_path = str(
                        hit.get("collision", hit.get("rigidBody", "")) or ""
                    )
                    if "/World/Characters/" in coll_path:
                        continue
                    # Floor / ground plane scrapes are not useful SFM obstacles.
                    if "Ground" in coll_path or "/ground" in coll_path.lower():
                        continue
                    distance = hit.get("distance", max_distance)
                    if distance < 0.15:
                        continue
                    hit_found = True
                    # Keep the closest hit
                    if distance < best_distance:
                        best_distance = distance
                        best_hit = hit.get("position")
            # If at least one hit was found, append the best hit; else, use defaults
            if hit_found:
                closest_hits.append((best_distance, best_hit))
            else:
                closest_hits.append((max_distance, None))

        return closest_hits

    def euler_from_quaternion(
        self, x: float, y: float, z: float, w: float
    ) -> Tuple[float, float, float]:
        """
        Converts a quaternion (with w as the scalar part) to Euler roll, pitch, yaw.
        quaternion = [x, y, z, w]
        """
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def send_agents_msg(self):
        """
        Called every physics step to call /compute_agents and update agent transforms.
        """
        if self.robot_prim is None:
            print("[HuNavManager] No robot assigned.")
            return

        # Build Agents message
        agents_msg = Agents()
        agents_msg.header = Header()
        now = self.node.get_clock().now().to_msg()
        agents_msg.header.stamp.sec = now.sec
        agents_msg.header.stamp.nanosec = now.nanosec
        agents_msg.header.frame_id = "world"

        # Build robot message
        robot_msg = self._create_robot_msg()

        # For each agent, create and add an Agent message
        for idx, agent_prim in enumerate(self.agents):
            agent_msg = self._create_agent_msg(agent_prim, idx)
            agents_msg.agents.append(agent_msg)

        # Wait for the compute_agents service and call it
        # PATCH (isaac-social-nav): first wait can be slow (loader spawn); do not
        # burn 2s on every physics tick after failure — use short polls.
        if not self.compute_agents_client.service_is_ready():
            if not self.compute_agents_client.wait_for_service(timeout_sec=0.0):
                if not getattr(self, "_compute_wait_logged", False):
                    print("[HuNavManager] waiting for /compute_agents …")
                    self._compute_wait_logged = True
                return
            print("[HuNavManager] /compute_agents ready.")
        self._call_compute(agents_msg, robot_msg)

    @staticmethod
    def _parse_behavior_type(raw) -> int:
        """Map YAML behavior.type (int or name) to hunav_msgs AgentBehavior constants."""
        name_map = {
            "regular": AgentBehavior.BEH_REGULAR,
            "impassive": AgentBehavior.BEH_IMPASSIVE,
            "surprised": AgentBehavior.BEH_SURPRISED,
            "scared": AgentBehavior.BEH_SCARED,
            "curious": AgentBehavior.BEH_CURIOUS,
            "threatening": AgentBehavior.BEH_THREATENING,
        }
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key.isdigit():
                return int(key)
            if key in name_map:
                return name_map[key]
        return AgentBehavior.BEH_REGULAR

    def _lookup_goal_cfg(self, goal_id, agent_cfg):
        """Resolve a goal id against global_goals or agent-local goal entries."""
        params = self.config["hunav_loader"]["ros__parameters"]
        global_goals = params.get("global_goals") or {}

        candidates = [goal_id]
        if not isinstance(goal_id, str):
            candidates.append(str(goal_id))
        else:
            try:
                candidates.append(int(goal_id))
            except ValueError:
                pass

        for key in candidates:
            if key in global_goals:
                return global_goals[key]
            if key in agent_cfg and isinstance(agent_cfg[key], dict) and "x" in agent_cfg[key]:
                return agent_cfg[key]
        return None

    def _resolve_agent_goals(self, agent_cfg, agent_name):
        """Build geometry_msgs/Pose[] goals for /compute_agents initialization."""
        if agent_name in self._planned_goals_cache:
            return list(self._planned_goals_cache[agent_name])

        key_cfgs = []
        for goal_id in agent_cfg.get("goals", []) or []:
            gcfg = self._lookup_goal_cfg(goal_id, agent_cfg)
            if gcfg is None:
                print(
                    f"[HuNavManager] {agent_name}: unknown goal id {goal_id!r}; skipping"
                )
                continue
            key_cfgs.append(gcfg)

        params = self.config["hunav_loader"]["ros__parameters"]
        planned_xy = None
        if self._occupancy_map is not None and key_cfgs:
            spacing = float(params.get("plan_waypoint_spacing_m", 1.0))
            keys = [(float(g["x"]), float(g["y"])) for g in key_cfgs]
            route_keys = list(keys)
            if params.get("plan_from_init_pose", True):
                init = agent_cfg.get("init_pose") or {}
                route_keys = [(float(init["x"]), float(init["y"]))] + route_keys
            # Close the loop for cyclic patrols so the return leg is also corridor-safe.
            if agent_cfg.get("cyclic_goals", False) and len(keys) >= 2:
                if route_keys[-1] != keys[0]:
                    route_keys = route_keys + [keys[0]]
            planned_xy = self._occupancy_map.plan_route(
                route_keys, waypoint_spacing_m=spacing
            )
            if planned_xy is None:
                print(
                    f"[HuNavManager] WARNING: {agent_name} occupancy plan failed; "
                    "falling back to sparse YAML goals (straight chords)."
                )
            else:
                # Drop the init-pose vertex — HuNav goals are destinations only.
                if params.get("plan_from_init_pose", True) and len(planned_xy) > 1:
                    planned_xy = planned_xy[1:]
                print(
                    f"[HuNavManager] {agent_name}: planned {len(planned_xy)} waypoints "
                    f"from {len(key_cfgs)} key goals (spacing={spacing} m)"
                )

        goals = []
        if planned_xy:
            for x, y in planned_xy:
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0
                goals.append(pose)
        else:
            for gcfg in key_cfgs:
                pose = Pose()
                pose.position.x = float(gcfg["x"])
                pose.position.y = float(gcfg["y"])
                pose.position.z = float(gcfg.get("z", 0.0))
                if "h" in gcfg:
                    q = Gf.Rotation(
                        Gf.Vec3d(0, 0, 1), float(gcfg["h"]) * 180.0 / math.pi
                    ).GetQuat()
                    pose.orientation.w = float(q.GetReal())
                    imag = q.GetImaginary()
                    pose.orientation.x = float(imag[0])
                    pose.orientation.y = float(imag[1])
                    pose.orientation.z = float(imag[2])
                goals.append(pose)

        if not goals:
            print(
                f"[HuNavManager] WARNING: {agent_name} has no resolvable goals; "
                "SFM will leave the agent idle."
            )
        self._planned_goals_cache[agent_name] = list(goals)
        return goals

    def _create_robot_msg(self):
        # Retrieve robot pose and velocities from the WheeledRobot object
        pos, quat = self.robot_obj.get_world_pose()
        lin_vel = self.robot_obj.get_linear_velocity()
        ang_vel = self.robot_obj.get_angular_velocity()

        robot = Agent()
        robot.id = 0
        robot.type = Agent.ROBOT
        robot.skin = 1
        robot.name = "Robot"
        robot.group_id = 0
        robot.radius = 0.5
        robot.desired_velocity = 1.0
        robot.linear_vel = np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2 + lin_vel[2] ** 2)
        robot.angular_vel = np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)

        # Pose
        robot.position.position.x = float(pos[0])
        robot.position.position.y = float(pos[1])
        robot.position.position.z = float(pos[2])
        robot.position.orientation.w = float(quat[0])
        robot.position.orientation.x = float(quat[1])
        robot.position.orientation.y = float(quat[2])
        robot.position.orientation.z = float(quat[3])
        _, _, yaw = self.euler_from_quaternion(quat[1], quat[2], quat[3], quat[0])
        robot.yaw = yaw

        # Velocities
        robot.velocity.linear.x = float(lin_vel[0])
        robot.velocity.linear.y = float(lin_vel[1])
        robot.velocity.linear.z = float(lin_vel[2])
        robot.velocity.angular.x = float(ang_vel[0])
        robot.velocity.angular.y = float(ang_vel[1])
        robot.velocity.angular.z = float(ang_vel[2])
        robot.cyclic_goals = True
        robot.goal_radius = 0.5
        robot.closest_obs = []
        return robot

    def _create_agent_msg(self, agent_prim, index):
        agent_ref = self.config["hunav_loader"]["ros__parameters"]["agents"][index]
        agent_cfg = self.config["hunav_loader"]["ros__parameters"][agent_ref]

        agent = Agent()
        agent.id = int(agent_cfg["id"])
        agent.type = Agent.PERSON
        if self.config["hunav_loader"]["ros__parameters"]["simulator"] == "Gazebo":
            agent.skin = agent_cfg["skin"]
        else:
            agent.skin = 0  
        agent.name = f"Agent{index + 1}"
        agent.group_id = int(agent_cfg["group_id"])
        agent.radius = float(agent_cfg["radius"])
        agent.desired_velocity = float(agent_cfg["max_vel"])

        # Read transforms
        pos = agent_prim.GetAttribute("xformOp:translate").Get()
        rot = agent_prim.GetAttribute("xformOp:orient").Get()
        rw = rot.GetReal()
        rx, ry, rz = rot.GetImaginary()

        # Position (pin Z to spawn height — SFM is planar)
        init_z = float(agent_cfg["init_pose"].get("z", 0.0))
        agent.position.position.x = float(pos[0])
        agent.position.position.y = float(pos[1])
        agent.position.position.z = init_z
        agent.position.orientation.x = float(rx)
        agent.position.orientation.y = float(ry)
        agent.position.orientation.z = float(rz)
        agent.position.orientation.w = float(rw)
        _, _, yaw = self.euler_from_quaternion(
            float(rx), float(ry), float(rz), float(rw)
        )
        agent.yaw = self.normalize_angle(yaw - math.pi / 2.0)

        # Velocities — SFM is planar; zero Z / roll-pitch junk from PhysX.
        lin = agent_prim.GetAttribute("physics:velocity").Get()
        ang = agent_prim.GetAttribute("physics:angularVelocity").Get()
        if lin is None:
            lin = Gf.Vec3d(0, 0, 0)
        if ang is None:
            ang = Gf.Vec3d(0, 0, 0)
        vx, vy = float(lin[0]), float(lin[1])
        agent.linear_vel = float(math.hypot(vx, vy))
        agent.angular_vel = float(ang[2])
        agent.velocity.linear.x = vx
        agent.velocity.linear.y = vy
        agent.velocity.linear.z = 0.0
        agent.velocity.angular.x = 0.0
        agent.velocity.angular.y = 0.0
        agent.velocity.angular.z = float(ang[2])

        # Goals — required by AgentManager.initializeAgents() for SFM attraction.
        # Upstream never filled these; without them agents idle in place.
        agent.cyclic_goals = agent_cfg["cyclic_goals"]
        agent.goal_radius = float(agent_cfg["goal_radius"])
        agent.goals = self._resolve_agent_goals(agent_cfg, agent.name)

        # Behavior
        beh = agent_cfg["behavior"]
        configuration = int(beh["configuration"])
        
        # SFM parameter configuration constants
        DEFAULT_SFM_PARAMS = {
            "goal_force_factor": 10.0,
            "obstacle_force_factor": 2.0,
            "social_force_factor": 5.0
        }
        
        SFM_CONSTRAINTS = {
            "goal_force_factor": (5.0, 10.0),
            "obstacle_force_factor": (0.5, 5.0),
            "social_force_factor": (5.0, 20.0)
        }
        
        def clamp_value(value: float, min_val: float, max_val: float) -> float:
            """Constrain value within specified range."""
            return max(min_val, min(value, max_val))
        
        # Set SFM parameters based on configuration type
        if configuration == 0:  # Default Isaac Sim configuration
            sfm_params = DEFAULT_SFM_PARAMS.copy()
            sfm_params["other_force_factor"] = float(beh["other_force_factor"])
        elif configuration == 1:  # Custom unconstrained configuration
            sfm_params = {
                "social_force_factor": float(beh["social_force_factor"]),
                "goal_force_factor": float(beh["goal_force_factor"]),
                "obstacle_force_factor": float(beh["obstacle_force_factor"]),
                "other_force_factor": float(beh["other_force_factor"])
            }
        else:  # Other configurations with constraints
            sfm_params = {
                "social_force_factor": float(beh["social_force_factor"]),
                "goal_force_factor": float(beh["goal_force_factor"]),
                "obstacle_force_factor": float(beh["obstacle_force_factor"]),
                "other_force_factor": float(beh["other_force_factor"])
            }
            
            # Apply constraints for non-custom configurations
            for param, (min_val, max_val) in SFM_CONSTRAINTS.items():
                if param in sfm_params:
                    sfm_params[param] = clamp_value(sfm_params[param], min_val, max_val)
        
        # state=0 idle; HuNav reaction fns set state=1 while reacting.
        # once/dist/duration come from YAML (Impassive has no once — default False).
        agent.behavior = AgentBehavior(
            type=self._parse_behavior_type(beh.get("type", 1)),
            state=0,
            configuration=configuration,
            duration=float(beh.get("duration", 40.0)),
            once=bool(beh.get("once", False)),
            vel=float(beh.get("vel", 0.6)),
            dist=float(beh.get("dist", 0.0)),
            social_force_factor=sfm_params["social_force_factor"],
            goal_force_factor=sfm_params["goal_force_factor"],
            obstacle_force_factor=sfm_params["obstacle_force_factor"],
            other_force_factor=sfm_params["other_force_factor"],
        )
        
        # Obstacle detection.
        # PATCH (isaac-social-nav): character meshes used to poison rays → spin.
        # get_closest_obstacles skips /World/Characters/*; prefer rays ON with
        # occupancy-planned goals. YAML ``ignore_obstacle_rays: true`` remains
        # an emergency opt-out (museum bring-up used it before map planning).
        agent.closest_obs = []
        params = self.config["hunav_loader"]["ros__parameters"]
        if params.get("ignore_obstacle_rays", False):
            for _ in range(90):
                agent.closest_obs.append(Point(x=10000.0, y=10000.0, z=10000.0))
            return agent

        max_distance = 4.0
        sensor_offsets = [0.9, 1.1, 1.3]  # torso heights; skip floor scrapes
        # Ray origins use pinned Z, not a fallen PhysX translate.
        ray_pos = Gf.Vec3d(float(pos[0]), float(pos[1]), init_z)
        hits = self.get_closest_obstacles(ray_pos, max_distance, sensor_offsets)
        for hit in hits:
            if hit[1] is not None:
                pt = Point(
                    x=float(hit[1][0]),
                    y=float(hit[1][1]),
                    z=float(hit[1][2]),
                )
            else:
                pt = Point(x=10000.0, y=10000.0, z=10000.0)
            agent.closest_obs.append(pt)
        return agent

    def _call_compute(self, agents_msg, robot_msg):
        try:
            req = ComputeAgents.Request()
            req.current_agents = agents_msg
            req.robot = robot_msg
            future = self.compute_agents_client.call_async(req)

            rclpy.spin_until_future_complete(self.node, future)
            if future.done():
                resp = future.result()
                if resp is None:
                    print("[HuNavManager] No response from service.")
                else:
                    self._update_agents(resp.updated_agents)
                return resp
            else:
                print("[HuNavManager] Service response not completed.")
                return None
        except Exception as e:
            print(f"[HuNavManager] Error calling service: {e}")
            return None

    def _update_agents(self, updated_agents):
        for upd in updated_agents.agents:
            idx = upd.id - 1
            agent_prim = self.agents[idx]
            agent_skelroot_prim = self.stage.GetPrimAtPath(
                agent_prim.GetPath().AppendChild("Animation")
            )

            anim_graph_path = self.bound_animations.get(agent_skelroot_prim.GetPath())

            if not self.flag_anim[agent_skelroot_prim.GetPath()]:
                self.stage.GetPrimAtPath(
                    find_skelroot_path(agent_skelroot_prim)
                ).SetMetadata("kind", "component")
                self.flag_anim[agent_skelroot_prim.GetPath()] = True

            char = ag.get_character(str(find_skelroot_path(agent_skelroot_prim)))

            # Pin Z to spawn height — SFM is 2D; PhysX/orientation noise must
            # not accumulate into vertical drift.
            agent_ref = self.config["hunav_loader"]["ros__parameters"]["agents"][idx]
            init_z = float(
                self.config["hunav_loader"]["ros__parameters"][agent_ref]["init_pose"].get(
                    "z", 0.0
                )
            )
            new_pos = Gf.Vec3d(
                upd.position.position.x,
                upd.position.position.y,
                init_z,
            )

            # Planar velocity only — never feed PhysX /people a huge Z component.
            lin = Gf.Vec3d(upd.velocity.linear.x, upd.velocity.linear.y, 0.0)
            speed_xy = float(math.hypot(lin[0], lin[1]))

            agent_path = agent_prim.GetPath()
            # PATCH (isaac-social-nav): do NOT rewrite near-robot XY/yaw here.
            # Fighting HuNav poses caused 180° flickers, mid-turn freezes, and
            # Curious "bumping". Behaviour fixes belong in hunav_agent_manager;
            # Isaac only applies poses + freezes yaw when nearly stopped.
            prev_xy = self.agent_previous_positions.get(agent_path)
            dist_robot = None
            rx = ry = None
            dx = dy = 0.0
            try:
                rpos, _ = self.robot_obj.get_world_pose()
                rx, ry = float(rpos[0]), float(rpos[1])
                dist_robot = math.hypot(
                    rx - float(new_pos[0]),
                    ry - float(new_pos[1]),
                )
            except Exception:
                dist_robot = None

            if prev_xy is not None:
                dx = float(new_pos[0]) - prev_xy[0]
                dy = float(new_pos[1]) - prev_xy[1]

            self.agent_previous_positions[agent_path] = (
                float(new_pos[0]),
                float(new_pos[1]),
            )
            agent_prim.GetAttribute("xformOp:translate").Set(new_pos)

            # HuNav orientation (tf2 RPY yaw) + character axis correction.
            new_quat = Gf.Quatf(
                upd.position.orientation.w,
                upd.position.orientation.x,
                upd.position.orientation.y,
                upd.position.orientation.z,
            )

            rotX = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90).GetQuat()
            rotXQ = Gf.Quatf(rotX)
            rotZ = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat()
            rotZQ = Gf.Quatf(rotZ)
            target_prim_quat = new_quat * rotXQ * rotZQ
            rotZQ_anim = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 0).GetQuat())
            target_anim_quat = new_quat * rotXQ * rotZQ_anim

            # Freeze yaw only when truly idle — raising this threshold trapped
            # Scared mid-turn. Slerp when moving.
            if speed_xy < 0.12 and agent_path in self.agent_previous_orientations:
                smoothed_prim_quat, smoothed_anim_quat = (
                    self.agent_previous_orientations[agent_path]
                )
            elif agent_path in self.agent_previous_orientations:
                prev_prim_quat, prev_anim_quat = self.agent_previous_orientations[
                    agent_path
                ]
                smoothed_prim_quat = self.slerp_quaternions(
                    prev_prim_quat,
                    target_prim_quat,
                    self.orientation_smoothing_factor,
                )
                smoothed_anim_quat = self.slerp_quaternions(
                    prev_anim_quat,
                    target_anim_quat,
                    self.orientation_smoothing_factor,
                )
            else:
                smoothed_prim_quat = target_prim_quat
                smoothed_anim_quat = target_anim_quat

            self.agent_previous_orientations[agent_path] = (
                smoothed_prim_quat,
                smoothed_anim_quat,
            )
            agent_prim.GetAttribute("xformOp:orient").Set(smoothed_prim_quat)

            # Animation orientation correction (use smoothed animation orientation)
            if char:
                pos_carb = carb.Float3(new_pos[0], new_pos[1], new_pos[2])
                real = smoothed_anim_quat.GetReal()
                imag = smoothed_anim_quat.GetImaginary()
                rot_carb = carb.Float4(imag[0], imag[1], imag[2], real)
                char.set_world_transform(pos_carb, rot_carb)
                # Character API can rewrite the container xform; re-pin XY/Z.
                agent_prim.GetAttribute("xformOp:translate").Set(new_pos)

            # Set velocities (yaw-rate only; kill tumble)
            agent_prim.GetAttribute("physics:velocity").Set(lin)
            ang = Gf.Vec3d(0.0, 0.0, float(upd.velocity.angular.z))
            agent_prim.GetAttribute("physics:angularVelocity").Set(ang)

            # Optional headless diagnosis: HUNAV_REACTION_LOG=/tmp/reaction.csv
            log_path = os.environ.get("HUNAV_REACTION_LOG", "").strip()
            if log_path:
                self._append_reaction_log(
                    log_path,
                    upd=upd,
                    x=float(new_pos[0]),
                    y=float(new_pos[1]),
                    speed_xy=speed_xy,
                    dx=dx,
                    dy=dy,
                    dist_robot=dist_robot,
                    yaw=float(getattr(upd, "yaw", 0.0) or 0.0),
                )

            # Set animation based on agent's planar speed
            speed = speed_xy
            max_expected_speed = 1.5
            normalized_speed = np.clip(speed / max_expected_speed, 0.0, 1.0)
            if anim_graph_path:
                set_anim_graph_speed(
                    self.stage, char, anim_graph_path, normalized_speed
                )
            else:
                print(f"No AnimationGraph bound for {agent_prim.GetPath()}")

            # Behavior name overlay (viewport DrawLabel).
            if self._behavior_labels.enabled:
                beh = getattr(upd, "behavior", None)
                beh_type = int(getattr(beh, "type", 0) or 0)
                beh_state = int(getattr(beh, "state", 0) or 0)
                self._behavior_labels.update_label(
                    int(upd.id),
                    float(new_pos[0]),
                    float(new_pos[1]),
                    float(new_pos[2]),
                    beh_type,
                    beh_state,
                )

    def _append_reaction_log(
        self,
        log_path,
        *,
        upd,
        x,
        y,
        speed_xy,
        dx,
        dy,
        dist_robot,
        yaw,
    ):
        """Append one CSV row for near-robot reaction diagnosis (headless)."""
        write_header = not os.path.isfile(log_path) or os.path.getsize(log_path) == 0
        try:
            beh = int(getattr(getattr(upd, "behavior", None), "type", 0) or 0)
            with open(log_path, "a", encoding="utf-8") as fh:
                if write_header:
                    fh.write(
                        "t_wall,agent_id,beh,x,y,yaw,speed_xy,dx,dy,dist_robot,"
                        "yaw_jump_deg\n"
                    )
                prev = self._reaction_log_prev.get(upd.id)
                yaw_jump = 0.0
                if prev is not None:
                    dyaw = yaw - prev["yaw"]
                    while dyaw > math.pi:
                        dyaw -= 2.0 * math.pi
                    while dyaw < -math.pi:
                        dyaw += 2.0 * math.pi
                    yaw_jump = abs(math.degrees(dyaw))
                self._reaction_log_prev[upd.id] = {"yaw": yaw}
                fh.write(
                    f"{time.time():.3f},{upd.id},{beh},{x:.4f},{y:.4f},{yaw:.4f},"
                    f"{speed_xy:.4f},{dx:.4f},{dy:.4f},"
                    f"{(dist_robot if dist_robot is not None else -1):.4f},"
                    f"{yaw_jump:.2f}\n"
                )
        except Exception as exc:
            # Logging must never break the sim loop.
            if not getattr(self, "_reaction_log_warned", False):
                print(f"[HuNavManager] reaction log failed: {exc}")
                self._reaction_log_warned = True

    def get_character_model_from_skin(self, skin_value):
        """
        Get character model path based on skin value.
        
        Args:
            skin_value (int): The skin ID from agent configuration
                            - 0: Random character model selection
                            - 1-11: Specific character models
            
        Returns:
            str: Path to the character model, or None if skin_value is invalid
        """
        # Handle random skin option (skin value 0)
        if skin_value == 0:
            random_index = random.randint(0, len(self.target_model_paths) - 1)
            return self.target_model_paths[random_index]
        
        # Handle specific skin values (1-11 mapped to model indices 0-10)
        if isinstance(skin_value, (int, float)) and skin_value in self.skin_to_model_mapping:
            model_index = self.skin_to_model_mapping[int(skin_value)]
            if 0 <= model_index < len(self.target_model_paths):
                return self.target_model_paths[model_index]
        
        return None

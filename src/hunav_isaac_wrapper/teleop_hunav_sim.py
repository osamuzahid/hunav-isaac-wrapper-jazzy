#!/usr/bin/env python3
"""
teleop_hunav_sim.py

Contains the TeleopHuNavSim class which combines:
- ROS 2 teleoperation for a differential robot.
- World loading via WorldBuilder.
- Agent management via HuNavManager.
"""
from isaacsim import SimulationApp

# ---------------------------------------------------------------------------
# ORIGINALLY (upstream v2.0): hard-coded SimulationApp CONFIG — kept for reference.
# Had to be patched because this laptop is under Isaac 6.0 minima (8GB VRAM / ~16GB
# RAM): always launching 1280x720 windowed RaytracedLighting OOMs / thrashes swap.
# ---------------------------------------------------------------------------
# CONFIG = {
#     "width": 1280,
#     "height": 720,
#     "sync_loads": True,
#     "headless": False,
#     "renderer": "RaytracedLighting",
# }
# simulation_app = SimulationApp(CONFIG)
# ---------------------------------------------------------------------------
# PATCH (isaac-social-nav): selectable profiles via sim_app_config.py
# (HUNAV_ISAAC_PROFILE / --profile / --debug). default|lab keeps the original
# 1280x720 windowed settings; debug|laptop uses 960x540 headless so bring-up
# and E2E smoke can run on this host. Also injects Kit --enable for anim graph.
# ---------------------------------------------------------------------------
from .sim_app_config import build_simulation_config, simulation_app_kwargs

_CONFIG_FULL = build_simulation_config()
CONFIG = simulation_app_kwargs(_CONFIG_FULL)
print(
    f"[hunav_isaac_wrapper] SimulationApp profile={_CONFIG_FULL.get('_profile')} "
    f"{CONFIG.get('width')}x{CONFIG.get('height')} headless={CONFIG.get('headless')} "
    f"renderer={CONFIG.get('renderer')}"
)
simulation_app = SimulationApp(CONFIG)

import os
import signal
import subprocess
import math
import yaml
import numpy as np
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
import omni
import omni.graph.core as og

# Import the WorldBuilder and HuNavManager modules.
from .world_builder import WorldBuilder
from .hunav_manager import HuNavManager

def find_package_share_directory():
    """
    Find the package share directory containing worlds, scenarios, config, etc.
    Works both in development and installed package modes.
    """
    # Prefer the source checkout. colcon --symlink-install links worlds/*.usd
    # into share/ but does not install nested assets/office, so opening the
    # install office.usd leaves furniture references unresolved (whitebox only).
    current_file = Path(__file__)
    if current_file.parent.parent.name == "src":
        src_dir = current_file.parent.parent
        if (src_dir / "worlds").exists():
            return str(src_dir)

    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        share_dir = pkg_path / "share" / "hunav_isaac_wrapper"
        if share_dir.exists() and (share_dir / "worlds").exists():
            return str(share_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    cwd = Path.cwd()
    if (cwd / "worlds").exists():
        return str(cwd)

    return os.path.dirname(os.path.dirname(__file__))


def find_robot_config_path(filename):
    """
    Find robot configuration file in development or installed package.
    
    Args:
        filename: Name of the robot config file (e.g., "nova_carter_ros2_sensors.usd")
    
    Returns:
        str: Absolute path to the robot config file
    """
    # Try to find via ROS2 package share directory (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        robot_config = pkg_path / "share" / "hunav_isaac_wrapper" / "config" / "robots" / filename
        if robot_config.exists():
            return str(robot_config)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try development mode (relative to this file)
    current_file_dir = Path(__file__).parent
    workspace_root = current_file_dir.parent.parent
    robot_config = workspace_root / "config" / "robots" / filename
    if robot_config.exists():
        return str(robot_config)
    
    # Try alternative development paths
    dev_paths = [
        current_file_dir.parent / "config" / "robots" / filename,
        Path.cwd() / "src" / "config" / "robots" / filename,
        Path.cwd() / "config" / "robots" / filename,
    ]
    
    for path in dev_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Robot config file not found: {filename}")

    raise FileNotFoundError(f"Robot config file not found: {filename}")


def _expand_nova_carter_body_instances(robot_root: str = "/World/Nova_Carter") -> int:
    """Un-instance Carter body/wheel visuals so they render and select as real meshes.

    Full_Merged ships chassis/wheel `visual` prims as USD instances. Selecting the
    empty `/World/Nova_Carter` wrapper then highlights a useless map-sized bound;
    expanding instances makes `chassis_link/visual/top_body` ordinary Mesh prims.
    Call after Configuration=Full_Merged and ideally before world.reset().
    """
    try:
        import omni.usd
        from pxr import Usd
    except Exception:
        return 0
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0
    root = stage.GetPrimAtPath(robot_root)
    if not root or not root.IsValid():
        return 0
    n = 0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath()).lower()
        if "/visual" not in path:
            continue
        if not (prim.IsInstanceable() or prim.IsInstance()):
            continue
        try:
            prim.SetInstanceable(False)
            n += 1
        except Exception:
            pass
    if n:
        print(
            f"[hunav_isaac_wrapper] expanded {n} Nova_Carter visual instance(s) "
            f"under {robot_root}",
            flush=True,
        )
    return n


def _ensure_nova_carter_visual_config(
    robot_prim_path: str, selection: str = "Full_Merged"
) -> str | None:
    """Nova Carter USD defaults to Skirt_only / No_Internals — nearly invisible body.

    Select Configuration=Full_Merged so chassis top_body + skirt + wheels exist,
    then expand instanceable visuals into real Mesh prims.
    Call after spawn and again after world.reset().
    """
    try:
        import omni.usd
    except Exception:
        return None
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    # Variant lives on the robot root after WheeledRobot flatten.
    candidates = [robot_prim_path]
    root = stage.GetPrimAtPath(robot_prim_path)
    if root and root.IsValid():
        for child in root.GetChildren():
            candidates.append(str(child.GetPath()))
    applied = None
    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        vsets = prim.GetVariantSets()
        if not vsets.HasVariantSet("Configuration"):
            continue
        vs = vsets.GetVariantSet("Configuration")
        opts = list(vs.GetVariantNames())
        sel = selection
        if sel not in opts:
            # Prefer a solid silhouette over Skirt_only.
            for fallback in ("Full_Merged", "No_Internals", "Base"):
                if fallback in opts:
                    sel = fallback
                    break
            else:
                continue
        prev = vs.GetVariantSelection()
        vs.SetVariantSelection(sel)
        print(
            f"[hunav_isaac_wrapper] Nova_Carter Configuration "
            f"{prev!r} → {sel!r} on {path}",
            flush=True,
        )
        applied = sel
        break
    if applied:
        _expand_nova_carter_body_instances(
            robot_prim_path
            if stage.GetPrimAtPath(robot_prim_path)
            else "/World/Nova_Carter"
        )
    return applied


def _load_robot_init_pose(hunav_config_path):
    """
    PATCH (isaac-social-nav): optional robot_init_pose from hunav scenario YAML.
    Returns dict with x,y,z,h (defaults 0).
    """
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "h": 0.0}
    if not hunav_config_path:
        return pose
    config_path = Path(hunav_config_path)
    if not config_path.is_file():
        return pose
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        init = data.get("hunav_loader", {}).get("ros__parameters", {}).get(
            "robot_init_pose", {}
        )
        if isinstance(init, dict):
            pose.update({k: float(init.get(k, pose[k])) for k in pose})
    except Exception as exc:
        print(f"[hunav_isaac_wrapper] WARNING: could not read robot_init_pose: {exc}")
    return pose


def _yaw_to_orientation(h):
    """Quaternion (w, x, y, z) from yaw about +Z."""
    return [math.cos(h / 2.0), 0.0, 0.0, math.sin(h / 2.0)]


def _yaw_from_orientation(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ChassisDriveRobot:
    """
    PATCH (isaac-social-nav): kinematic planar base for Stretch and Reachy.
    Spawns a vendored USD and integrates /cmd_vel on the root Xform each step.
    Uses Physics variant ``none`` so arm/lift joints are not PhysX-simulated
    (Stretch otherwise collapses; Reachy omniwheel PhysX flings the arm).
    """

    def __init__(self, prim_path, usd_path, position, orientation):
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            vsets = prim.GetVariantSets()
            if vsets.HasVariantSet("Physics"):
                # Keep mesh hierarchy, drop ArticulationRoot / rigid bodies.
                vsets.GetVariantSet("Physics").SetVariantSelection("none")
                print(
                    f"[ChassisDriveRobot] {prim_path}: Physics variant → none "
                    "(kinematic visual; no joint collapse)"
                )
        self._xform = SingleXFormPrim(
            prim_path=prim_path,
            name="Robot",
            position=position,
            orientation=orientation,
        )
        self._lin_vel = np.zeros(3, dtype=float)
        self._ang_vel = np.zeros(3, dtype=float)

    def apply_cmd_vel(self, lin_x, ang_z, dt):
        pos, quat = self._xform.get_world_pose()
        yaw = _yaw_from_orientation(quat)
        yaw += ang_z * dt
        pos = np.asarray(pos, dtype=float)
        pos[0] += lin_x * math.cos(yaw) * dt
        pos[1] += lin_x * math.sin(yaw) * dt
        new_quat = np.asarray(_yaw_to_orientation(yaw), dtype=float)
        self._xform.set_world_pose(position=pos, orientation=new_quat)
        self._lin_vel = np.array(
            [lin_x * math.cos(yaw), lin_x * math.sin(yaw), 0.0], dtype=float
        )
        self._ang_vel = np.array([0.0, 0.0, ang_z], dtype=float)

    def get_world_pose(self):
        return self._xform.get_world_pose()

    def get_linear_velocity(self):
        return self._lin_vel.copy()

    def get_angular_velocity(self):
        return self._ang_vel.copy()


class StaticPrimRobot:
    """
    PATCH (isaac-social-nav): parked lab robot (Franka first; Reachy/A1 later).
    Spawns a USD at robot_init_pose; no /cmd_vel drive. HuNav still reads pose.
    Optional USD variant selections (e.g. Franka Gripper/Mesh).
    """

    def __init__(
        self,
        prim_path,
        usd_path,
        position,
        orientation,
        variants=None,
        articulation_prim=None,
        expand_instances=False,
        park_kinematic=False,
    ):
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid() and variants:
            vsets = prim.GetVariantSets()
            for vset_name, selection in variants.items():
                if not vsets.HasVariantSet(vset_name):
                    continue
                vs = vsets.GetVariantSet(vset_name)
                if selection not in vs.GetVariantNames():
                    print(
                        f"[StaticPrimRobot] {prim_path}: variant "
                        f"{vset_name}={selection!r} not in {list(vs.GetVariantNames())}"
                    )
                    continue
                prev = vs.GetVariantSelection()
                vs.SetVariantSelection(selection)
                print(
                    f"[StaticPrimRobot] {prim_path}: {vset_name} "
                    f"{prev!r} → {selection!r}"
                )
        if expand_instances and prim and prim.IsValid():
            from pxr import Usd

            n = 0
            for p in Usd.PrimRange(prim):
                if p.IsInstanceable():
                    p.SetInstanceable(False)
                    n += 1
            if n:
                print(f"[StaticPrimRobot] {prim_path}: expanded {n} instanceable prims")
        # Parked lab morphologies: freeze PhysX so bad URDF inertias / zero-stiffness
        # drives cannot fling links apart (Reachy shoulder_x had ixx~1e4).
        if park_kinematic and prim and prim.IsValid():
            from pxr import Usd, UsdPhysics

            n_kin = 0
            for p in Usd.PrimRange(prim):
                if not p.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rb = UsdPhysics.RigidBodyAPI(p)
                attr = rb.GetKinematicEnabledAttr()
                if not attr:
                    attr = rb.CreateKinematicEnabledAttr(True)
                else:
                    attr.Set(True)
                n_kin += 1
            print(
                f"[StaticPrimRobot] {prim_path}: kinematic park on {n_kin} rigid bodies"
            )
        self._xform = SingleXFormPrim(
            prim_path=prim_path,
            name="Robot",
            position=position,
            orientation=orientation,
        )
        self._lin_vel = np.zeros(3, dtype=float)
        self._ang_vel = np.zeros(3, dtype=float)
        self._articulation = None
        self._held_joint_positions = None
        self.prim_path = prim_path
        # Optional child path with ArticulationRoot (Reachy: Geometry/world).
        self._articulation_prim = articulation_prim or prim_path

    def try_init_articulation(self):
        """After world.reset(), hold joints at the spawned / zero pose."""
        try:
            from isaacsim.core.prims import SingleArticulation
        except Exception as exc:
            print(f"[StaticPrimRobot] SingleArticulation import failed: {exc}")
            return False
        candidates = [self._articulation_prim, self.prim_path]
        # Deduplicate while preserving order.
        seen = set()
        paths = []
        for p in candidates:
            if p and p not in seen:
                seen.add(p)
                paths.append(p)
        last_exc = None
        for path in paths:
            try:
                art = SingleArticulation(path)
                art.initialize()
                self._articulation = art
                pos = np.array(art.get_joint_positions(), dtype=float, copy=True)
                # Park at zeros when DOFs are near-zero / unset.
                if np.allclose(pos, 0.0, atol=1e-3) or not np.isfinite(pos).all():
                    pos = np.zeros_like(pos)
                else:
                    # Still force a quiet park for lab demos.
                    pos = np.zeros_like(pos)
                art.set_joint_positions(pos)
                self._held_joint_positions = pos
                print(
                    f"[StaticPrimRobot] articulation OK at {path} "
                    f"({len(self._held_joint_positions)} DOFs held at zero)"
                )
                return True
            except Exception as exc:
                last_exc = exc
                continue
        print(f"[StaticPrimRobot] articulation init skipped: {last_exc}")
        self._articulation = None
        self._held_joint_positions = None
        return False

    def hold_joints(self):
        if self._articulation is None or self._held_joint_positions is None:
            return
        try:
            self._articulation.set_joint_positions(self._held_joint_positions)
        except Exception:
            pass

    def get_world_pose(self):
        return self._xform.get_world_pose()

    def get_linear_velocity(self):
        return self._lin_vel.copy()

    def get_angular_velocity(self):
        return self._ang_vel.copy()


class TeleopHuNavSim(Node):
    """
    Combines:
    - Differential robot teleop (subscribing to /cmd_vel)
    - USD map loading (via WorldBuilder)
    - Agent management and update (via HuNavManager)
    """

    def __init__(self, map_name, hunav_config, robot_name):
        super().__init__("hunav_sim")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Assets root
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Nucleus root.")

        # Load USD stage
        self.builder = WorldBuilder(base_path=find_package_share_directory())
        if map_name:
            self.builder.load_map(map_name)

        # Create World object
        timestep = 1.0 / 20.0
        self.world = World(
            stage_units_in_meters=1, physics_dt=timestep, rendering_dt=timestep
        )

        if map_name == "empty_world":
            self.world.scene.add_default_ground_plane()

        # ---------------------------------------------------------------------------
        # ORIGINALLY (upstream v2.0) — Isaac 4.5/5.x robot USD layout. Kept for reference.
        # Had to be patched because on Isaac 6.0 CDN these paths return HTTP 404
        # (robots moved under NVIDIA/ / iRobot/Create3/; Carter sensors USD gone;
        # carter_ROS zip not extracted by default).
        # ---------------------------------------------------------------------------
        # robot_configs = {
        #     "jetbot": {
        #         "name": "Jetbot",
        #         "usd_relative_path": os.path.join(
        #             "Isaac", "Robots", "Jetbot", "jetbot.usd"
        #         ),
        #         "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
        #         "wheel_radius": 0.0325,
        #         "wheel_base": 0.118,
        #     },
        #     "create3": {
        #         "name": "Create3",
        #         "usd_relative_path": os.path.join(
        #             "Isaac", "Robots", "iRobot", "create_3.usd"
        #         ),
        #         "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
        #         "wheel_radius": 0.03575,
        #         "wheel_base": 0.233,
        #     },
        #     "carter": {
        #         "name": "Nova_Carter",
        #         "usd_relative_path": os.path.join(
        #             "Isaac", "Robots", "Carter", "nova_carter_sensors.usd"
        #         ),
        #         "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
        #         "wheel_radius": 0.14,
        #         "wheel_base": 0.413,
        #     },
        #     "carter_ROS": {
        #         "name": "Nova_Carter",
        #         "usd_relative_path": find_robot_config_path(
        #             "nova_carter_ros2_sensors.usd"
        #         ),
        #         "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
        #         "wheel_radius": 0.14,
        #         "wheel_base": 0.413,
        #     },
        # }
        # ---------------------------------------------------------------------------
        # PATCH (isaac-social-nav): remap usd_relative_path to Isaac 6.0 CDN locations
        # verified HTTP 200 + omni.client.stat OK. Wheel DOFs / radii unchanged.
        # Important so robots actually spawn instead of failing asset resolve.
        # ---------------------------------------------------------------------------
        robot_configs = {
            "jetbot": {
                "name": "Jetbot",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "NVIDIA", "Jetbot", "jetbot.usd"
                ),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.0325,
                "wheel_base": 0.118,
            },
            "create3": {
                "name": "Create3",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "iRobot", "Create3", "create_3.usd"
                ),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.03575,
                "wheel_base": 0.233,
            },
            "carter": {
                "name": "Nova_Carter",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "NVIDIA", "NovaCarter", "nova_carter.usd"
                ),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
                # PATCH: CUCR floors need clearance (same class of issue as Stretch).
                "spawn_z_lift": 0.08,
            },
            "carter_ROS": {
                "name": "Nova_Carter",
                "usd_relative_path": os.path.join(
                    "Isaac", "Samples", "ROS2", "Robots", "Nova_Carter_ROS.usd"
                ),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
                "spawn_z_lift": 0.08,
            },
        }
        # PATCH (isaac-social-nav): Stretch / Reachy (and later lab robots) live in
        # config/robots/<name>/robot.yaml — do not add them to the dict above.
        from .robot_catalog import load_lab_spawn_configs

        robot_configs.update(load_lab_spawn_configs())

        if robot_name not in robot_configs:
            raise ValueError(f"Unsupported robot_name: {robot_name}")

        robot_config = dict(robot_configs[robot_name])
        if "usd_package_file" in robot_config:
            robot_config["usd_relative_path"] = find_robot_config_path(
                robot_config["usd_package_file"]
            )
        robot_init = _load_robot_init_pose(hunav_config)
        spawn_z = robot_init["z"] + float(robot_config.get("spawn_z_lift", 0.0))
        spawn_position = [
            robot_init["x"],
            robot_init["y"],
            spawn_z,
        ]
        spawn_orientation = _yaw_to_orientation(robot_init["h"])
        
        # Handle absolute vs relative paths for robot USD files
        if os.path.isabs(robot_config["usd_relative_path"]):
            # Absolute path (for custom robots like carter_ROS)
            robot_path = robot_config["usd_relative_path"]
        else:
            # Relative path (for built-in Isaac Sim robots)
            robot_path = os.path.join(assets_root_path, robot_config["usd_relative_path"])

        # Add robot to world
        robot_prim_path = f"/World/{robot_config['name']}"
        self._robot_name = robot_name
        self._lab_sensor_handles = None
        # ORIGINALLY (upstream v2.0): always WheeledRobot + DifferentialController at [0,0,0].
        # PATCH (isaac-social-nav): lab YAML `kind: chassis_only` / `static` / `differential`.
        drive = robot_config.get("drive")
        if drive == "chassis_only":
            self._chassis_drive = True
            self._static_drive = False
            self.robot = ChassisDriveRobot(
                prim_path=robot_prim_path,
                usd_path=robot_path,
                position=spawn_position,
                orientation=spawn_orientation,
            )
            self.diff_controller = None
            self._hold_non_wheel_dofs = False
            self._held_joint_positions = None
            self._wheel_dof_indices = None
        elif drive == "static":
            self._chassis_drive = False
            self._static_drive = True
            art_rel = robot_config.get("articulation_prim")
            art_prim = (
                f"{robot_prim_path}/{art_rel}" if art_rel else robot_prim_path
            )
            self.robot = StaticPrimRobot(
                prim_path=robot_prim_path,
                usd_path=robot_path,
                position=spawn_position,
                orientation=spawn_orientation,
                variants=robot_config.get("variants"),
                articulation_prim=art_prim,
                expand_instances=bool(robot_config.get("expand_instances")),
                park_kinematic=bool(robot_config.get("park_kinematic")),
            )
            self.diff_controller = None
            self._hold_non_wheel_dofs = False
            self._held_joint_positions = None
            self._wheel_dof_indices = None
        else:
            self._chassis_drive = False
            self._static_drive = False
            self.robot = self.world.scene.add(
                WheeledRobot(
                    prim_path=robot_prim_path,
                    name="Robot",
                    wheel_dof_names=robot_config["wheel_dof_names"],
                    create_robot=True,
                    usd_path=robot_path,
                    position=spawn_position,
                    orientation=spawn_orientation,
                )
            )

            # Create differential drive controller
            self.diff_controller = DifferentialController(
                name="diff_drive_controller",
                wheel_radius=robot_config["wheel_radius"],
                wheel_base=robot_config["wheel_base"],
            )
            # PATCH (isaac-social-nav): Stretch articulates lift/arm/head/gripper. Only the
            # two wheel DOFs are driven; unheld joints fall under gravity and the mesh
            # looks collapsed / detached from the root gizmo. Hold non-wheel DOFs at the
            # post-reset pose every control step.
            self._hold_non_wheel_dofs = bool(
                robot_config.get("hold_non_wheel_dofs", False)
            )
            self._held_joint_positions = None
            self._wheel_dof_indices = None

        # PATCH: Carter ROS USD ships Configuration=Skirt_only — body almost invisible.
        if robot_name in ("carter", "carter_ROS"):
            _ensure_nova_carter_visual_config(robot_prim_path, "Full_Merged")

        # ROS2 cmd_vel subscriber (BEST_EFFORT — matches ESC / Nav2 velocity QoS).
        self.cmd_lin = 0.00
        self.cmd_ang = 0.00
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, cmd_vel_qos
        )

        # Setup HuNavManager
        self.hunav = HuNavManager(
            node=self,
            world=self.world,
            config_file_path=hunav_config,
            robot_prim_path=robot_prim_path,
            robot=self.robot,
        )

        self.create_ros_clock_action_graph()

        self.hunav.initialize_agents()
        self.hunav.initialize_hunav_nodes()

        # PATCH (isaac-social-nav): stock lab-robot sensors (TF / joints / lidar / IMU).
        # Attached after agents so stage prims exist; cameras default off (HUNAV_LAB_CAMERAS).
        try:
            from .lab_robot_sensors import attach_lab_robot_sensors

            self._lab_sensor_handles = attach_lab_robot_sensors(
                robot_name, robot_prim_path, ros_node=self
            )
        except Exception as exc:
            print(f"[hunav_isaac_wrapper] lab sensors attach failed: {exc}")
            self._lab_sensor_handles = None

    def _signal_handler(self, signum, frame):
        print("\n\nCaught shutdown signal, closing app and stopping hunav nodes...\n\n")
        self.hunav.close_hunav_nodes()
        simulation_app.close()

    def _cmd_vel_callback(self, msg):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def _on_physics_step(self, dt: float):
        """
        Called automatically by PhysX each physics frame.
        """
        self.hunav.send_agents_msg()

    def create_ros_clock_action_graph(self, graph_path="/World/ROS2"):
        try:
            keys = og.Controller.Keys
            graph_params = {
                # Create the necessary nodes.
                keys.CREATE_NODES: [
                    # Node for generating a tick on playback.
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    # Node for reading the simulation time.
                    (
                        "isaac_read_simulation_time",
                        "isaacsim.core.nodes.IsaacReadSimulationTime",
                    ),
                    # Node to create a ROS2 context.
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    # Node to publish the clock over ROS2.
                    ("ros2_publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                # Connect outputs to inputs:
                keys.CONNECT: [
                    # Connect context output to the publish clock's context input.
                    (
                        "ros2_context.outputs:context",
                        "ros2_publish_clock.inputs:context",
                    ),
                    # Connect tick output to publish clock's execIn.
                    (
                        "on_playback_tick.outputs:tick",
                        "ros2_publish_clock.inputs:execIn",
                    ),
                    # Connect simulation time output to publish clock's timeStamp.
                    (
                        "isaac_read_simulation_time.outputs:simulationTime",
                        "ros2_publish_clock.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    # For the simulation time node.
                    # ORIGINALLY also set inputs:swhFrameNumber=0 — removed on Isaac 6.0
                    # (attr gone; only resetOnStop / referenceTime* remain).
                    ("isaac_read_simulation_time.inputs:resetOnStop", True),
                    # For the ROS2PublishClock node.
                    ("ros2_publish_clock.inputs:nodeNamespace", ""),
                    ("ros2_publish_clock.inputs:qosProfile", ""),
                    ("ros2_publish_clock.inputs:queueSize", 10),
                    ("ros2_publish_clock.inputs:timeStamp", 0.0),
                    ("ros2_publish_clock.inputs:topicName", "clock"),
                    # For the ROS2Context node.
                    ("ros2_context.inputs:useDomainIDEnvVar", True),
                    ("ros2_context.inputs:domain_id", 0),
                ],
            }
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                graph_params,
            )
            print(f"Successfully created ROS_Clock action graph at {graph_path}")
        except Exception as e:
            print(f"Error creating ROS_Clock action graph: {e}")

    def run(self):
        self.world.reset()
        # Stretch RGB: world.reset() clears camera child xforms; re-apply fixed
        # optical→OpenGL mount (not the world-up solve — that raced reset).
        if self._lab_sensor_handles:
            try:
                from .lab_robot_sensors import refresh_optical_camera_orients

                refresh_optical_camera_orients(self._lab_sensor_handles)
            except Exception as exc:
                print(f"[hunav_isaac_wrapper] camera orient refresh failed: {exc}")
        # ---------------------------------------------------------------------------
        # ORIGINALLY (upstream v2.0):
        # self.physx_interface = omni.physx.acquire_physx_interface()
        # Had to be patched because Isaac 6.0 renamed the API to get_physx_interface
        # (AttributeError: module 'omni.physx' has no attribute 'acquire_physx_interface').
        # ---------------------------------------------------------------------------
        # PATCH (isaac-social-nav): prefer get_physx_interface, fall back to acquire_*
        # so physics step subscription works on Isaac 6.0 (and older if present).
        # ---------------------------------------------------------------------------
        _physx = getattr(omni.physx, "get_physx_interface", None) or getattr(
            omni.physx, "acquire_physx_interface", None
        )
        self.physx_interface = _physx()
        self.physx_sub = self.physx_interface.subscribe_physics_step_events(
            self._on_physics_step
        )
        # Capture default Stretch pose after reset; re-apply non-wheel DOFs every step.
        if self._hold_non_wheel_dofs:
            names = list(self.robot.dof_names)
            self._wheel_dof_indices = {
                names.index("joint_left_wheel"),
                names.index("joint_right_wheel"),
            }
            self._held_joint_positions = np.array(
                self.robot.get_joint_positions(), dtype=float, copy=True
            )
        if self._static_drive and hasattr(self.robot, "try_init_articulation"):
            self.robot.try_init_articulation()
        self.hunav.send_agents_msg()
        while simulation_app.is_running():
            self.world.step(render=True)
            if self._lab_sensor_handles:
                try:
                    from .lab_robot_sensors import tick_lab_sensor_handles

                    tick_lab_sensor_handles(
                        self._lab_sensor_handles,
                        sim_time=float(self.world.current_time),
                    )
                except Exception:
                    pass
            if self._chassis_drive:
                dt = self.world.get_physics_dt()
                self.robot.apply_cmd_vel(self.cmd_lin, self.cmd_ang, dt)
            elif self._static_drive:
                if hasattr(self.robot, "hold_joints"):
                    self.robot.hold_joints()
            else:
                if self._hold_non_wheel_dofs and self._held_joint_positions is not None:
                    current = np.array(
                        self.robot.get_joint_positions(), dtype=float, copy=True
                    )
                    held = self._held_joint_positions.copy()
                    for wi in self._wheel_dof_indices:
                        held[wi] = current[wi]
                    self.robot.set_joint_positions(held)
                wheel_action = self.diff_controller.forward([self.cmd_lin, self.cmd_ang])
                self.robot.apply_wheel_actions(wheel_action)

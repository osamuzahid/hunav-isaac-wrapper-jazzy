"""
lab_robot_sensors.py

PATCH (isaac-social-nav): attach stock sensors for lab robots declared in
config/robots/<name>/robot.yaml (see robot_catalog.py). URDF→USD is morphology
only; lidar/cameras/IMU are Isaac prims under the YAML link paths.

Env:
  HUNAV_LAB_SENSORS=0|1   — master switch (default 1 when YAML has sensors:)
  HUNAV_LAB_CAMERAS=0|1   — RGB / RGB-D (default 0; heavy on under-spec laptops)
  HUNAV_LAB_LIDAR=0|1     — RTX 2D lidar (default 1)

Link paths, lidar TF pins, and parked joints are in robot.yaml — do not add a
new `if robot_name ==` branch here for a new lab robot.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import omni.graph.core as og


def _orient_opengl_camera_on_optical(stage, cam_prim, optical_path: str) -> None:
    """UsdGeom.Camera under ROS optical: look along +Z_opt, roll leveled to world +Z.

    OpenGL cameras look along local −Z with +Y up. Solve one local quaternion from
    world bases (no Euler patch stack). Look = optical +Z; up = world +Z projected
    orthogonal to look (so rqt is upright even when optical −Y is horizontal).

    WARNING: only valid once optical's world xform is final. Stretch attach runs
    *before* world.reset(), which clears child xforms — using this at attach (or
    immediately after reset before xforms evaluate) locks identity under optical
    → OpenGL looks along optical −Z = into the RealSense housing (top-down self-view).
    Prefer _orient_opengl_camera_stretch_fixed for Stretch.
    """
    from pxr import Gf, Usd, UsdGeom

    opt = stage.GetPrimAtPath(optical_path)
    if not opt or not opt.IsValid():
        raise RuntimeError(f"missing optical prim {optical_path}")
    mw = UsdGeom.Xformable(opt).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    forward = Gf.Vec3d(mw.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)))
    if forward.GetLength() < 1e-8:
        raise RuntimeError(f"optical +Z degenerate at {optical_path}")
    forward.Normalize()

    world_up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(forward, world_up)
    if right.GetLength() < 1e-3:
        # Nearly vertical look: fall back to optical −Y as the up hint.
        hint = Gf.Vec3d(mw.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)))
        right = Gf.Cross(forward, hint)
    right.Normalize()
    up = Gf.Cross(right, forward)
    up.Normalize()

    # Row-major: columns are camera axes in world (X=right, Y=up, Z=−look).
    r_cam = Gf.Matrix3d(
        right[0],
        up[0],
        -forward[0],
        right[1],
        up[1],
        -forward[1],
        right[2],
        up[2],
        -forward[2],
    )
    r_local = mw.ExtractRotationMatrix().GetInverse() * r_cam
    quat = Gf.Quatf(r_local.ExtractRotation().GetQuat())

    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat)


def _orient_opengl_camera_fixed_optical(cam_prim) -> None:
    """Fixed optical→OpenGL: 180° about +X (Reachy; avoids attach-time world-pose race)."""
    from pxr import Gf, UsdGeom

    quat = Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat)


def _orient_opengl_camera_stretch_fixed(cam_prim) -> None:
    """Stretch: fixed optical→OpenGL, no world-pose dependency.

    Rx(180) maps OpenGL −Z ← optical +Z (RealSense look). On parked Stretch,
    optical −Y is world −Y so Rx(180) alone is sideways in rqt; Rz(−90) about
    the look axis levels camera +Y to world +Z (USD rest pose probe).
    Immune to world.reset() clearing xforms / attach-before-reset race.
    """
    import math

    from pxr import Gf, UsdGeom

    # Gf.Quatf(real, i, j, k). Apply Rx then Rz in parent (optical) frame.
    rx = Gf.Quatf(0.0, 1.0, 0.0, 0.0)  # 180° about X
    hs = math.sin(-math.pi / 4.0)
    hc = math.cos(-math.pi / 4.0)
    rz = Gf.Quatf(hc, 0.0, 0.0, hs)  # −90° about Z
    quat = Gf.Quatf(rz * rx)
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat)

# Parked Stretch joint names (kinematic Physics=none has no articulation).
_STRETCH_PARKED_JOINTS = [
    "joint_left_wheel",
    "joint_right_wheel",
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_arm_l4",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_gripper_finger_left",
    "joint_gripper_finger_right",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def lab_sensors_enabled(robot_name: str) -> bool:
    from .robot_catalog import load_lab_robot_yaml

    spec = load_lab_robot_yaml(robot_name)
    if not spec or not spec.get("sensors"):
        return False
    return _env_bool("HUNAV_LAB_SENSORS", True)


def _prim_exists(path: str) -> bool:
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        prim = stage.GetPrimAtPath(path)
        return bool(prim and prim.IsValid())
    except Exception:
        return False


def _join(root: str, rel: str) -> str:
    return f"{root.rstrip('/')}/{rel.lstrip('/')}"


def _read_laser_scan_metadata(prim) -> Dict[str, Any]:
    rotation_rate = float(prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get() or 0)
    near_range = float(prim.GetAttribute("omni:sensor:Core:nearRangeM").Get() or 0)
    far_range = float(prim.GetAttribute("omni:sensor:Core:farRangeM").Get() or 0)
    firing_rate = int(prim.GetAttribute("omni:sensor:Core:patternFiringRateHz").Get() or 0)
    if rotation_rate <= 0 or firing_rate <= 0:
        # Safe defaults matching Example_Rotary_2D-ish behavior.
        return {
            "horizontalFov": 360.0,
            "horizontalResolution": 1.0,
            "depthRange": [0.1, 30.0],
            "rotationRate": 10.0,
            "azimuthRange": [-180.0, 180.0],
        }
    return {
        "horizontalFov": 360.0,
        "horizontalResolution": 360.0 * rotation_rate / firing_rate,
        "depthRange": [near_range, far_range],
        "rotationRate": rotation_rate,
        "azimuthRange": [-180.0, 180.0],
    }


def _attach_tf_tree(graph_path: str, target_prims: Sequence[str], topic: str = "tf") -> None:
    import usdrt.Sdf

    keys = og.Controller.Keys
    paths = [usdrt.Sdf.Path(p) for p in target_prims if _prim_exists(p)]
    if not paths:
        print(f"[lab_robot_sensors] TF skip: no valid target prims for {graph_path}")
        return
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishTF.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    ("PublishTF.inputs:topicName", topic),
                    ("PublishTF.inputs:targetPrims", paths),
                ],
            },
        )
        print(f"[lab_robot_sensors] TF graph {graph_path} → /{topic} ({len(paths)} prims)")
    except Exception as exc:
        print(f"[lab_robot_sensors] TF graph failed: {exc}")


def _attach_joint_state_publisher(
    graph_path: str, articulation_prim: str, topic: str = "joint_states"
) -> bool:
    import usdrt.Sdf

    if not _prim_exists(articulation_prim):
        print(
            f"[lab_robot_sensors] joint_states skip: missing prim {articulation_prim}"
        )
        return False
    keys = og.Controller.Keys
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
                    (
                        "ReadJointState.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "ReadJointState.outputs:jointNames",
                        "PublishJointState.inputs:jointNames",
                    ),
                    (
                        "ReadJointState.outputs:jointPositions",
                        "PublishJointState.inputs:jointPositions",
                    ),
                    (
                        "ReadJointState.outputs:jointVelocities",
                        "PublishJointState.inputs:jointVelocities",
                    ),
                    (
                        "ReadJointState.outputs:jointEfforts",
                        "PublishJointState.inputs:jointEfforts",
                    ),
                    (
                        "ReadJointState.outputs:jointDofTypes",
                        "PublishJointState.inputs:jointDofTypes",
                    ),
                    (
                        "ReadJointState.outputs:stageMetersPerUnit",
                        "PublishJointState.inputs:stageMetersPerUnit",
                    ),
                    (
                        "ReadJointState.outputs:sensorTime",
                        "PublishJointState.inputs:sensorTime",
                    ),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                ],
                keys.SET_VALUES: [
                    (
                        "ReadJointState.inputs:prim",
                        [usdrt.Sdf.Path(articulation_prim)],
                    ),
                    ("PublishJointState.inputs:topicName", topic),
                    ("Context.inputs:useDomainIDEnvVar", True),
                ],
            },
        )
        print(
            f"[lab_robot_sensors] joint_states graph {graph_path} "
            f"prim={articulation_prim} → /{topic}"
        )
        return True
    except Exception as exc:
        print(f"[lab_robot_sensors] joint_states graph failed: {exc}")
        return False


def _level_rtx_2d_lidar_beam(lidar_prim) -> None:
    """Set Example_Rotary_2D emitter elevation to 0° (horizontal).

    NVIDIA's stock ``Example_Rotary_2D`` profile aims rays at elevationDeg=[-2].
    A real Stretch SE3 RPLidar scans in the horizontal plane of the ``laser``
    frame. Override the OmniLidar emitter-state attribute — do not pitch the
    lidar prim to cancel the example tilt.
    """
    changed = []
    for attr in lidar_prim.GetAttributes():
        name = attr.GetName()
        if "emitterState" not in name or not name.endswith(":elevationDeg"):
            continue
        prev = attr.Get()
        n = len(list(prev)) if prev is not None else 1
        attr.Set([0.0] * n)
        changed.append(f"{name} (was {list(prev) if prev is not None else None})")
    if changed:
        print(
            "[lab_robot_sensors] lidar beam leveled to 0° (RPLidar-horizontal): "
            + "; ".join(changed)
        )
    else:
        print(
            "[lab_robot_sensors] WARNING: no emitterState:*/elevationDeg on lidar "
            "prim — stock −2° tilt may remain"
        )


def _attach_stretch_lidar(laser_prim: str, frame_id: str = "laser") -> Optional[Any]:
    if not _env_bool("HUNAV_LAB_LIDAR", True):
        print("[lab_robot_sensors] lidar disabled (HUNAV_LAB_LIDAR=0)")
        return None
    if not _prim_exists(laser_prim):
        print(f"[lab_robot_sensors] lidar skip: missing {laser_prim}")
        return None
    try:
        import isaacsim.core.experimental.utils.prim as prim_utils
        from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

        lidar_path = f"{laser_prim}/rtx_lidar"
        # Base NVIDIA 2D rotary asset at the URDF ``laser`` mount. Elevation is
        # corrected after create (see ``_level_rtx_2d_lidar_beam``).
        lidar = Lidar.create(
            path=lidar_path,
            config="Example_Rotary_2D",
            tick_rate=10.0,
            translations=[[0.0, 0.0, 0.0]],
        )
        _level_rtx_2d_lidar_beam(lidar.prims[0])
        # NVIDIA requires tickRate == scanRateBaseHz for a full rotary scan per
        # published LaserScan (stock Example_Rotary_2D is 30 Hz).
        prim = lidar.prims[0]
        scan_hz = prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get()
        if scan_hz is not None and float(scan_hz) > 0:
            tick_attr = prim.GetAttribute("omni:sensor:tickRate")
            if tick_attr and tick_attr.IsValid():
                prev = tick_attr.Get()
                tick_attr.Set(float(scan_hz))
                print(
                    f"[lab_robot_sensors] lidar tickRate {prev} → {scan_hz} "
                    "(match scanRateBaseHz)"
                )
        sensor = LidarSensor(lidar, annotators=[])
        meta = _read_laser_scan_metadata(prim_utils.get_prim_at_path(lidar.paths[0]))
        sensor.attach_writer(
            "RtxLidarROS2PublishLaserScan",
            topicName="scan",
            frameId=frame_id,
            **meta,
        )
        print(f"[lab_robot_sensors] RTX 2D lidar at {lidar.paths[0]} → /scan")
        return sensor
    except Exception as exc:
        print(f"[lab_robot_sensors] lidar attach failed: {exc}")
        return None


def _attach_rgb_optical_camera(
    optical_prim: str,
    *,
    graph_path: str,
    topic_ns: str,
    frame_id: str,
    width: int = 320,
    height: int = 240,
    with_depth: bool = False,
    fixed_optical_orient: bool = False,
) -> bool:
    """Mount OpenGL Camera under a ROS optical frame and publish RGB (+ optional depth)."""
    if not _env_bool("HUNAV_LAB_CAMERAS", False):
        print("[lab_robot_sensors] cameras off (set HUNAV_LAB_CAMERAS=1 to enable)")
        return False
    if not _prim_exists(optical_prim):
        print(f"[lab_robot_sensors] camera skip: missing {optical_prim}")
        return False
    try:
        import omni.usd
        import usdrt.Sdf
        from pxr import Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_path = f"{optical_prim}/rgb_camera"
        if stage.GetPrimAtPath(cam_path):
            stage.RemovePrim(cam_path)
        cam = UsdGeom.Camera(stage.DefinePrim(cam_path, "Camera"))
        if fixed_optical_orient:
            _orient_opengl_camera_fixed_optical(cam.GetPrim())
        else:
            _orient_opengl_camera_on_optical(stage, cam.GetPrim(), optical_prim)
        cam.GetHorizontalApertureAttr().Set(21)
        cam.GetVerticalApertureAttr().Set(16)
        cam.GetProjectionAttr().Set("perspective")
        cam.GetFocalLengthAttr().Set(24)
        cam.GetPrim().CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(0.02)

        keys = og.Controller.Keys
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        connect = [
            ("OnPlaybackTick.outputs:tick", "createRenderProduct.inputs:execIn"),
            ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            (
                "createRenderProduct.outputs:renderProductPath",
                "cameraHelperRgb.inputs:renderProductPath",
            ),
            (
                "createRenderProduct.outputs:renderProductPath",
                "cameraHelperInfo.inputs:renderProductPath",
            ),
        ]
        set_values = [
            ("createRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
            ("createRenderProduct.inputs:width", width),
            ("createRenderProduct.inputs:height", height),
            ("cameraHelperRgb.inputs:frameId", frame_id),
            ("cameraHelperRgb.inputs:topicName", f"{topic_ns}/image_raw"),
            ("cameraHelperRgb.inputs:type", "rgb"),
            ("cameraHelperInfo.inputs:frameId", frame_id),
            ("cameraHelperInfo.inputs:topicName", f"{topic_ns}/camera_info"),
        ]
        if with_depth:
            create_nodes.append(
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper")
            )
            connect.extend(
                [
                    (
                        "createRenderProduct.outputs:execOut",
                        "cameraHelperDepth.inputs:execIn",
                    ),
                    (
                        "createRenderProduct.outputs:renderProductPath",
                        "cameraHelperDepth.inputs:renderProductPath",
                    ),
                ]
            )
            set_values.extend(
                [
                    ("cameraHelperDepth.inputs:frameId", frame_id),
                    ("cameraHelperDepth.inputs:topicName", f"{topic_ns}/depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ]
            )

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: create_nodes,
                keys.CONNECT: connect,
                keys.SET_VALUES: set_values,
            },
        )
        print(
            f"[lab_robot_sensors] RGB camera at {cam_path} "
            f"({width}x{height}) → /{topic_ns}/* (frame={frame_id})"
        )
        return True
    except Exception as exc:
        print(f"[lab_robot_sensors] camera attach failed ({topic_ns}): {exc}")
        return False


def _attach_stretch_camera(
    optical_prim: str, width: int = 320, height: int = 240
) -> Optional[Dict[str, str]]:
    # Keep Stretch RGB-D path (depth + legacy topic names).
    if not _env_bool("HUNAV_LAB_CAMERAS", False):
        print("[lab_robot_sensors] cameras off (set HUNAV_LAB_CAMERAS=1 to enable)")
        return None
    if not _prim_exists(optical_prim):
        print(f"[lab_robot_sensors] camera skip: missing {optical_prim}")
        return None
    try:
        import omni.usd
        import usdrt.Sdf
        from pxr import Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        link_prim = optical_prim.rsplit("/camera_color_frame", 1)[0]
        for stale in (
            f"{link_prim}/rgb_camera",
            f"{optical_prim}/rgb_camera",
        ):
            if stage.GetPrimAtPath(stale):
                stage.RemovePrim(stale)

        cam_path = f"{optical_prim}/rgb_camera"
        cam = UsdGeom.Camera(stage.DefinePrim(cam_path, "Camera"))
        # Fixed optical→OpenGL (no world matrix). World-up solve at attach/reset
        # raced world.reset() and produced identity-under-optical = head self-view.
        _orient_opengl_camera_stretch_fixed(cam.GetPrim())
        cam.GetHorizontalApertureAttr().Set(21)
        cam.GetVerticalApertureAttr().Set(16)
        cam.GetProjectionAttr().Set("perspective")
        cam.GetFocalLengthAttr().Set(24)
        cam.GetPrim().CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(0.02)

        keys = og.Controller.Keys
        og.Controller.edit(
            {
                "graph_path": "/World/ROS2_LabCamera",
                "evaluator_name": "execution",
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "createRenderProduct",
                        "isaacsim.core.nodes.IsaacCreateRenderProduct",
                    ),
                    ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "createRenderProduct.inputs:execIn",
                    ),
                    (
                        "createRenderProduct.outputs:execOut",
                        "cameraHelperRgb.inputs:execIn",
                    ),
                    (
                        "createRenderProduct.outputs:execOut",
                        "cameraHelperInfo.inputs:execIn",
                    ),
                    (
                        "createRenderProduct.outputs:execOut",
                        "cameraHelperDepth.inputs:execIn",
                    ),
                    (
                        "createRenderProduct.outputs:renderProductPath",
                        "cameraHelperRgb.inputs:renderProductPath",
                    ),
                    (
                        "createRenderProduct.outputs:renderProductPath",
                        "cameraHelperInfo.inputs:renderProductPath",
                    ),
                    (
                        "createRenderProduct.outputs:renderProductPath",
                        "cameraHelperDepth.inputs:renderProductPath",
                    ),
                ],
                keys.SET_VALUES: [
                    (
                        "createRenderProduct.inputs:cameraPrim",
                        [usdrt.Sdf.Path(cam_path)],
                    ),
                    ("createRenderProduct.inputs:width", width),
                    ("createRenderProduct.inputs:height", height),
                    ("cameraHelperRgb.inputs:frameId", "camera_color_optical_frame"),
                    ("cameraHelperRgb.inputs:topicName", "camera/color/image_raw"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperInfo.inputs:frameId", "camera_color_optical_frame"),
                    ("cameraHelperInfo.inputs:topicName", "camera/color/camera_info"),
                    ("cameraHelperDepth.inputs:frameId", "camera_color_optical_frame"),
                    (
                        "cameraHelperDepth.inputs:topicName",
                        "camera/depth/image_rect_raw",
                    ),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ],
            },
        )
        print(
            f"[lab_robot_sensors] RGB-D camera at {cam_path} "
            f"({width}x{height}) → /camera/color/* /camera/depth/* "
            "(fixed optical Rx180·Rz-90; reset-safe)"
        )
        return {"cam_path": cam_path, "optical_path": optical_prim}
    except Exception as exc:
        print(f"[lab_robot_sensors] camera attach failed: {exc}")
        return None


def refresh_optical_camera_orients(handles: Optional[Dict[str, Any]]) -> None:
    """Re-apply Stretch fixed optical mount after world.reset() clears child xforms."""
    if not handles:
        return
    specs = handles.get("optical_cam_refresh") or []
    if not specs:
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for spec in specs:
        cam_path = spec.get("cam_path")
        if not cam_path:
            continue
        cam = stage.GetPrimAtPath(cam_path)
        if not cam or not cam.IsValid():
            continue
        try:
            _orient_opengl_camera_stretch_fixed(cam)
            print(
                f"[lab_robot_sensors] re-applied Stretch fixed camera orient on {cam_path} "
                "(post-reset)"
            )
        except Exception as exc:
            print(f"[lab_robot_sensors] camera refresh failed ({cam_path}): {exc}")


def _attach_synthetic_imu_publisher(
    graph_path: str, frame_id: str = "base_imu", topic: str = "imu"
) -> None:
    """Publish gravity-only IMU for static / kinematic bases (no PhysX body needed)."""
    keys = og.Controller.Keys
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishImu.inputs:execIn"),
                    ("Context.outputs:context", "PublishImu.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishImu.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    ("PublishImu.inputs:topicName", topic),
                    ("PublishImu.inputs:frameId", frame_id),
                    ("PublishImu.inputs:linearAcceleration", [0.0, 0.0, 9.81]),
                    ("PublishImu.inputs:angularVelocity", [0.0, 0.0, 0.0]),
                    ("PublishImu.inputs:orientation", [0.0, 0.0, 0.0, 1.0]),
                    ("Context.inputs:useDomainIDEnvVar", True),
                ],
            },
        )
        print(
            f"[lab_robot_sensors] synthetic IMU → /{topic} "
            f"(frame={frame_id}, gravity-only)"
        )
    except Exception as exc:
        print(f"[lab_robot_sensors] IMU publish failed: {exc}")


class ParkedJointStatePublisher:
    """rclpy publisher of fixed joint_states for kinematic Stretch (Physics=none)."""

    def __init__(self, node, joint_names: Optional[Sequence[str]] = None):
        from sensor_msgs.msg import JointState

        self._JointState = JointState
        self._names = list(joint_names or _STRETCH_PARKED_JOINTS)
        self._pub = node.create_publisher(JointState, "/joint_states", 10)
        self._zeros = [0.0] * len(self._names)

    def publish(self, stamp_sec: float = 0.0) -> None:
        msg = self._JointState()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec % 1.0) * 1e9)
        msg.name = list(self._names)
        msg.position = list(self._zeros)
        msg.velocity = list(self._zeros)
        msg.effort = list(self._zeros)
        self._pub.publish(msg)


class ParkedLinkTfPublisher:
    """rclpy /tf from USD Xform world poses (PoseTree needs RigidBody; opticals don't)."""

    def __init__(
        self,
        node,
        frames: Sequence[tuple],
        parent_frame: str = "world",
    ):
        """
        frames: sequence of (prim_path, child_frame_id).
        Each pose is published as parent_frame → child_frame_id.
        """
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )

        self._TransformStamped = TransformStamped
        self._TFMessage = TFMessage
        self._parent = parent_frame
        self._frames = [(p, f) for p, f in frames if _prim_exists(p)]
        # RELIABLE is what RViz / tf2 expect. BEST_EFFORT subscribers (Isaac
        # smoke) still receive RELIABLE publishers; the reverse does not work.
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._pub = node.create_publisher(TFMessage, "/tf", tf_qos)
        self._logged_ok = False
        self._logged_err = False
        if not self._frames:
            print("[lab_robot_sensors] parked TF: no valid prims")
        else:
            print(
                f"[lab_robot_sensors] parked TF frames: "
                + ", ".join(f for _, f in self._frames)
            )

    def publish(self, stamp_sec: float = 0.0) -> None:
        if not self._frames:
            return
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            sec = int(stamp_sec)
            nsec = int((stamp_sec % 1.0) * 1e9)
            out = self._TFMessage()
            for prim_path, frame_id in self._frames:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                mw = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                t = mw.ExtractTranslation()
                q = mw.ExtractRotation().GetQuat()
                imag = q.GetImaginary()
                ts = self._TransformStamped()
                ts.header.stamp.sec = sec
                ts.header.stamp.nanosec = nsec
                ts.header.frame_id = self._parent
                ts.child_frame_id = frame_id
                ts.transform.translation.x = float(t[0])
                ts.transform.translation.y = float(t[1])
                ts.transform.translation.z = float(t[2])
                ts.transform.rotation.x = float(imag[0])
                ts.transform.rotation.y = float(imag[1])
                ts.transform.rotation.z = float(imag[2])
                ts.transform.rotation.w = float(q.GetReal())
                out.transforms.append(ts)
            if out.transforms:
                self._pub.publish(out)
                if not self._logged_ok:
                    print(
                        f"[lab_robot_sensors] parked TF publishing "
                        f"{len(out.transforms)} transforms"
                    )
                    self._logged_ok = True
        except Exception as exc:
            if not self._logged_err:
                print(f"[lab_robot_sensors] parked TF publish failed: {exc}")
                self._logged_err = True


def _gf_pose(mw):
    """USD world matrix → (xyz, xyzw quat)."""
    t = mw.ExtractTranslation()
    q = mw.ExtractRotation().GetQuat()
    imag = q.GetImaginary()
    return (
        (float(t[0]), float(t[1]), float(t[2])),
        (float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())),
    )


def _relative_gf(mw_parent, mw_child):
    """Child pose in the parent frame.

    USD Gf is row-vector (p' = p * M). ORIGINALLY ``inverse(parent) * child``
    (column-vector) put Stretch ``laser`` ~10–18 m off the chassis, so Nav2's
    local costmap logged ``Sensor origin out of map bounds`` and skipped
    /scan.
    """
    from pxr import Gf

    rel = Gf.Matrix4d(mw_child) * Gf.Matrix4d(mw_parent).GetInverse()
    return _gf_pose(rel)


def _yaw_from_xyzw(x, y, z, w) -> float:
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class KinematicNavPublisher:
    """Nav2 TF tree + /odom for kinematic Stretch (Physics=none).

    ORIGINALLY ParkedLinkTfPublisher posted world→each link (forest). Nav2 needs
    map→odom→base_link→laser. World XY already matches the occupancy map, so
    map→odom is identity; odom is ground-truth chassis pose. world→map identity
    keeps RViz Fixed Frame ``world`` / ``laser`` working.

    /tf stays RELIABLE (RViz/tf2). /odom is BEST_EFFORT (Nav2 sensor QoS).
    Do not also publish world→base_link (two parents).
    """

    def __init__(
        self,
        node,
        base_prim: str,
        child_frames: Sequence[tuple],
        odom_topic: str = "/odom",
        fixed_in_base: Optional[Dict[str, tuple]] = None,
    ):
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )

        self._TransformStamped = TransformStamped
        self._TFMessage = TFMessage
        self._Odometry = Odometry
        self._base_prim = base_prim
        self._children = [(p, f) for p, f in child_frames if _prim_exists(p)]
        self._fixed_in_base = dict(fixed_in_base or {})
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._tf_pub = node.create_publisher(TFMessage, "/tf", tf_qos)
        self._odom_pub = node.create_publisher(Odometry, odom_topic, odom_qos)
        self._prev_xy = None
        self._prev_yaw = None
        self._prev_t = None
        self._logged_ok = False
        self._logged_err = False
        print(
            f"[lab_robot_sensors] kinematic Nav TF+odom: base={base_prim} "
            f"children={', '.join(f for _, f in self._children) or '(none)'} "
            f"topic={odom_topic}"
        )

    def _ts(self, stamp_sec, parent, child, xyz, xyzw):
        ts = self._TransformStamped()
        ts.header.stamp.sec = int(stamp_sec)
        ts.header.stamp.nanosec = int((stamp_sec % 1.0) * 1e9)
        ts.header.frame_id = parent
        ts.child_frame_id = child
        ts.transform.translation.x = float(xyz[0])
        ts.transform.translation.y = float(xyz[1])
        ts.transform.translation.z = float(xyz[2])
        ts.transform.rotation.x = float(xyzw[0])
        ts.transform.rotation.y = float(xyzw[1])
        ts.transform.rotation.z = float(xyzw[2])
        ts.transform.rotation.w = float(xyzw[3])
        return ts

    def publish(self, stamp_sec: float = 0.0) -> None:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            base = stage.GetPrimAtPath(self._base_prim)
            if not base or not base.IsValid():
                return
            mw_base = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            xyz, xyzw = _gf_pose(mw_base)
            ident = (0.0, 0.0, 0.0)
            iq = (0.0, 0.0, 0.0, 1.0)
            out = self._TFMessage()
            out.transforms.append(self._ts(stamp_sec, "world", "map", ident, iq))
            out.transforms.append(self._ts(stamp_sec, "map", "odom", ident, iq))
            out.transforms.append(
                self._ts(stamp_sec, "odom", "base_link", xyz, xyzw)
            )
            for prim_path, frame_id in self._children:
                if frame_id in self._fixed_in_base:
                    cxyz, cxyzw = self._fixed_in_base[frame_id]
                else:
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim or not prim.IsValid():
                        continue
                    mw = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    cxyz, cxyzw = _relative_gf(mw_base, mw)
                out.transforms.append(
                    self._ts(stamp_sec, "base_link", frame_id, cxyz, cxyzw)
                )
                if not self._logged_ok and frame_id in ("laser", "lidar_link"):
                    off = math.hypot(cxyz[0], cxyz[1])
                    print(
                        f"[lab_robot_sensors] {frame_id} in base_link "
                        f"({cxyz[0]:.3f},{cxyz[1]:.3f},{cxyz[2]:.3f}) "
                        f"xy={off:.3f} m (URDF mount pin)"
                    )
                    if off > 1.0:
                        print(
                            "[lab_robot_sensors] WARNING: lidar XY offset "
                            f"{off:.2f} m — local costmap will not raytrace"
                        )
            self._tf_pub.publish(out)

            yaw = _yaw_from_xyzw(*xyzw)
            vx = vy = wz = 0.0
            if self._prev_xy is not None and self._prev_t is not None:
                dt = float(stamp_sec) - float(self._prev_t)
                if dt > 1e-4:
                    dx = xyz[0] - self._prev_xy[0]
                    dy = xyz[1] - self._prev_xy[1]
                    vx_w = dx / dt
                    vy_w = dy / dt
                    c, s = math.cos(yaw), math.sin(yaw)
                    vx = vx_w * c + vy_w * s
                    vy = -vx_w * s + vy_w * c
                    dyaw = yaw - self._prev_yaw
                    while dyaw > math.pi:
                        dyaw -= 2.0 * math.pi
                    while dyaw < -math.pi:
                        dyaw += 2.0 * math.pi
                    wz = dyaw / dt
            self._prev_xy = (xyz[0], xyz[1])
            self._prev_yaw = yaw
            self._prev_t = float(stamp_sec)

            odom = self._Odometry()
            odom.header.stamp.sec = int(stamp_sec)
            odom.header.stamp.nanosec = int((stamp_sec % 1.0) * 1e9)
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = xyz[0]
            odom.pose.pose.position.y = xyz[1]
            odom.pose.pose.position.z = xyz[2]
            odom.pose.pose.orientation.x = xyzw[0]
            odom.pose.pose.orientation.y = xyzw[1]
            odom.pose.pose.orientation.z = xyzw[2]
            odom.pose.pose.orientation.w = xyzw[3]
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = wz
            self._odom_pub.publish(odom)

            if not self._logged_ok:
                print(
                    f"[lab_robot_sensors] kinematic Nav TF+odom live "
                    f"base=({xyz[0]:.2f},{xyz[1]:.2f})"
                )
                self._logged_ok = True
        except Exception as exc:
            if not self._logged_err:
                print(f"[lab_robot_sensors] kinematic Nav TF+odom failed: {exc}")
                self._logged_err = True


def attach_lab_robot_sensors(
    robot_name: str,
    robot_prim_path: str,
    ros_node=None,
) -> Dict[str, Any]:
    """
    Attach stock sensors from config/robots/<name>/robot.yaml.
    """
    from .robot_catalog import load_lab_robot_yaml

    handles: Dict[str, Any] = {"robot": robot_name, "prim": robot_prim_path}
    if not lab_sensors_enabled(robot_name):
        print(f"[lab_robot_sensors] skipped for {robot_name}")
        return handles

    spec = load_lab_robot_yaml(robot_name) or {}
    sensors = spec.get("sensors") or {}
    links = sensors.get("links") or {}
    resolved = {key: _join(robot_prim_path, rel) for key, rel in links.items()}
    joints = sensors.get("parked_joints")
    mode = sensors.get("mode", "kinematic_nav")
    base = resolved.get("base", robot_prim_path)

    if mode == "pose_tree":
        tf_targets = [
            p
            for p in [robot_prim_path, *resolved.values()]
            if _prim_exists(p)
        ]
        _attach_tf_tree("/World/ROS2_LabTF", tf_targets or [robot_prim_path])
        ok = _attach_joint_state_publisher("/World/ROS2_LabJoints", robot_prim_path)
        if not ok and ros_node is not None:
            handles["parked_js"] = ParkedJointStatePublisher(
                ros_node, joint_names=joints
            )
    elif mode == "kinematic_nav" and ros_node is not None:
        handles["parked_js"] = ParkedJointStatePublisher(
            ros_node, joint_names=joints
        )
        nav = sensors.get("nav_tf") or {}
        child_frames = []
        for item in nav.get("children") or []:
            prim = resolved.get(item["link"])
            if prim:
                child_frames.append((prim, item["frame_id"]))
        fixed_in_base = {}
        for frame_id, pose in (nav.get("fixed_in_base") or {}).items():
            fixed_in_base[frame_id] = (
                tuple(pose["xyz"]),
                tuple(pose["xyzw"]),
            )
        handles["parked_tf"] = KinematicNavPublisher(
            ros_node,
            base_prim=base,
            child_frames=child_frames,
            fixed_in_base=fixed_in_base,
        )
        print(
            f"[lab_robot_sensors] {robot_name} kinematic: /joint_states + Nav2 "
            "TF (map/odom/base_link) + /odom via rclpy (Physics=none; no PoseTree)"
        )

    lidar = sensors.get("lidar") or {}
    lidar_link = lidar.get("link")
    if lidar_link and lidar_link in resolved:
        handles["lidar"] = _attach_stretch_lidar(
            resolved[lidar_link],
            frame_id=str(lidar.get("frame_id", "laser")),
        )

    imu = sensors.get("imu") or {}
    if imu.get("frame_id"):
        _attach_synthetic_imu_publisher(
            str(imu.get("graph_path", "/World/ROS2_LabImu")),
            frame_id=str(imu["frame_id"]),
        )

    for cam in sensors.get("cameras") or []:
        optical_key = cam.get("optical")
        optical_prim = resolved.get(optical_key) if optical_key else None
        if not optical_prim:
            continue
        ctype = cam.get("type")
        if ctype == "stretch_rgbd":
            cam_spec = _attach_stretch_camera(optical_prim)
            if cam_spec:
                handles.setdefault("optical_cam_refresh", []).append(cam_spec)
        elif ctype == "rgb_optical":
            _attach_rgb_optical_camera(
                optical_prim,
                graph_path=str(cam.get("graph_path", "/World/ROS2_LabRgb")),
                topic_ns=str(cam.get("topic_ns", "camera")),
                frame_id=str(cam.get("frame_id", "camera_optical")),
                fixed_optical_orient=bool(cam.get("fixed_optical_orient", False)),
            )

    print(f"[lab_robot_sensors] {robot_name}: sensors from robot.yaml")
    return handles


def tick_lab_sensor_handles(handles: Optional[Dict[str, Any]], sim_time: float = 0.0) -> None:
    if not handles:
        return
    parked = handles.get("parked_js")
    if parked is not None:
        parked.publish(sim_time)
    parked_tf = handles.get("parked_tf")
    if parked_tf is not None:
        parked_tf.publish(sim_time)

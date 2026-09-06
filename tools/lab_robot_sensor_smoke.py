#!/usr/bin/env python3
"""
lab_robot_sensor_smoke.py

PATCH (isaac-social-nav): headless smoke for Phase 0 lab robots.

Spawns empty_world (or museum/hospital) with --robot stretch|reachy, steps sim,
and checks ROS topics **plus** message content for TF / joint_states / IMU
(and Stretch /scan hit count).

Usage (from wrapper repo root):
  source /opt/ros/jazzy/setup.bash
  source ../../../install/setup.bash   # ros2_ws
  export OMNI_KIT_ACCEPT_EULA=YES ROS_DOMAIN_ID=0 HUNAV_ISAAC_PROFILE=debug
  # Optional: HUNAV_LAB_CAMERAS=1  HUNAV_LAB_LIDAR=1
  ~/isaacsim/python.sh tools/lab_robot_sensor_smoke.py \\
    --robot stretch --world empty_world --seconds 50 \\
    --summary /tmp/lab_robot_stretch_summary.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Key Stretch parked joints (must appear on /joint_states).
_STRETCH_JS_REQUIRED = (
    "joint_lift",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_left_wheel",
    "joint_right_wheel",
)

# TF leaf names for Stretch Nav2 tree (map→odom→base_link→laser).
_STRETCH_TF_LEAVES = (
    "base_link",
    "odom",
    "laser",
    "base_imu",
)

# Key Reachy parked joints (must appear on /joint_states).
_REACHY_JS_REQUIRED = (
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
    "r_shoulder_pitch",
    "l_shoulder_pitch",
)

# TF child/parent leaf names expected for Reachy stock mounts (Nav2 tree).
_REACHY_TF_LEAVES = (
    "base_link",
    "odom",
    "lidar_link",
    "torso",
    "left_camera_optical",
    "right_camera_optical",
)


def _finite(xs) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in xs)
    except Exception:
        return False


def _frame_leaf(name: str) -> str:
    return str(name).rstrip("/").split("/")[-1]


def _external_content_capture(robot: str, out_json: str, wait_s: float = 12.0) -> Optional[Dict[str, Any]]:
    """
    Capture TF / joint_states / imu / scan from a *separate* ROS process.

    Isaac bridge pubs often do not deliver to same-process rclpy subscribers
    (and may use BEST_EFFORT). External shell is the reliable check path.
    """
    import json
    import subprocess
    import textwrap

    py = textwrap.dedent(
        f"""
        import json, math, time, sys
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy, HistoryPolicy, QoSProfile,
            ReliabilityPolicy, qos_profile_sensor_data,
        )
        from sensor_msgs.msg import Imu, JointState, LaserScan
        from tf2_msgs.msg import TFMessage

        robot = {robot!r}
        out = {out_json!r}
        deadline = time.time() + {float(wait_s)!r}

        class P(Node):
            def __init__(self):
                super().__init__('lab_smoke_ext')
                self.js = self.imu = self.scan = None
                self.frames = set()
                self.tf_finite = False
                tf_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=50,
                )
                self.create_subscription(JointState, '/joint_states', self._js, 10)
                self.create_subscription(
                    JointState, '/joint_states', self._js,
                    QoSProfile(
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10,
                    ),
                )
                self.create_subscription(TFMessage, '/tf', self._tf, tf_qos)
                if robot in ('stretch', 'stretch_wheeled'):
                    self.create_subscription(Imu, '/imu', self._imu, qos_profile_sensor_data)
                    self.create_subscription(LaserScan, '/scan', self._scan, qos_profile_sensor_data)

            def _js(self, m):
                self.js = m
            def _imu(self, m):
                self.imu = m
            def _scan(self, m):
                self.scan = m
            def _tf(self, m):
                for t in m.transforms:
                    for n in (t.header.frame_id, t.child_frame_id):
                        self.frames.add(str(n).rstrip('/').split('/')[-1])
                    tr = t.transform.translation
                    if all(math.isfinite(float(v)) for v in (tr.x, tr.y, tr.z)):
                        self.tf_finite = True

        rclpy.init()
        n = P()
        while time.time() < deadline:
            rclpy.spin_once(n, timeout_sec=0.1)
            done = n.js is not None and n.frames
            if robot == 'reachy':
                need = {{'base_link', 'odom', 'lidar_link'}}
                done = n.js is not None and need.issubset(n.frames)
                done = done and n.imu is not None and n.scan is not None
            elif robot in ('stretch', 'stretch_wheeled'):
                done = done and n.imu is not None and n.scan is not None
            if done:
                break
        data = {{
            'js_names': list(n.js.name) if n.js else [],
            'js_pos': [float(x) for x in (n.js.position or [])] if n.js else [],
            'tf_frames': sorted(n.frames),
            'tf_finite': bool(n.tf_finite),
            'imu': None,
            'scan_beams': 0,
            'scan_hits': 0,
            'scan_frame': '',
        }}
        if n.imu is not None:
            a, w = n.imu.linear_acceleration, n.imu.angular_velocity
            data['imu'] = {{
                'frame': str(n.imu.header.frame_id),
                'a': [float(a.x), float(a.y), float(a.z)],
                'w': [float(w.x), float(w.y), float(w.z)],
            }}
        if n.scan is not None:
            rs = list(n.scan.ranges or [])
            data['scan_beams'] = len(rs)
            data['scan_frame'] = str(n.scan.header.frame_id)
            data['scan_hits'] = sum(
                1 for r in rs
                if math.isfinite(r) and n.scan.range_min < r < n.scan.range_max
            )
        with open(out, 'w') as f:
            json.dump(data, f)
        n.destroy_node()
        rclpy.shutdown()
        """
    )
    try:
        # Prefer system ROS Python (separate from Isaac's interpreter).
        cmd = [
            "/bin/bash",
            "-lc",
            "source /opt/ros/jazzy/setup.bash && "
            f"export ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} && "
            "python3 -",
        ]
        subprocess.run(
            cmd,
            input=py,
            text=True,
            timeout=float(wait_s) + 15.0,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.isfile(out_json):
            with open(out_json, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        print(f"[lab_robot_smoke] external content capture failed: {exc}", flush=True)
    return None


def _evaluate_captured(robot: str, data: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not data:
        lines.append("content: FAIL (external capture empty)")
        return lines

    names = list(data.get("js_names") or [])
    pos = list(data.get("js_pos") or [])
    if not names:
        lines.append("content /joint_states: FAIL (no message)")
    elif pos and len(pos) != len(names):
        lines.append(
            f"content /joint_states: FAIL (len name={len(names)} pos={len(pos)})"
        )
    elif pos and not _finite(pos):
        lines.append("content /joint_states: FAIL (non-finite position)")
    elif robot in ("stretch", "stretch_wheeled"):
        missing = [j for j in _STRETCH_JS_REQUIRED if j not in names]
        if missing:
            lines.append(f"content /joint_states: FAIL (missing {missing})")
        else:
            lines.append(
                f"content /joint_states: PASS (n={len(names)}, parked keys present)"
            )
    elif robot == "reachy":
        missing = [j for j in _REACHY_JS_REQUIRED if j not in names]
        if missing:
            lines.append(f"content /joint_states: FAIL (missing {missing})")
        else:
            lines.append(
                f"content /joint_states: PASS (n={len(names)}, reachy keys present)"
            )
    else:
        if len(names) < 5:
            lines.append(f"content /joint_states: FAIL (only {len(names)} joints)")
        else:
            lines.append(f"content /joint_states: PASS (n={len(names)})")

    frames = set(data.get("tf_frames") or [])
    if not frames:
        lines.append("content /tf: FAIL (no transforms)")
    elif not data.get("tf_finite"):
        lines.append("content /tf: FAIL (non-finite translation)")
    elif robot in ("stretch", "stretch_wheeled"):
        missing = [f for f in _STRETCH_TF_LEAVES if f not in frames]
        if missing:
            lines.append(
                f"content /tf: FAIL (missing {missing}; saw {sorted(frames)[:12]})"
            )
        else:
            lines.append(
                f"content /tf: PASS "
                f"(base_link/laser/base_imu; {len(frames)} frames)"
            )
    elif robot == "reachy":
        required = ("base_link", "odom", "lidar_link")
        missing = [f for f in required if f not in frames]
        if missing:
            lines.append(
                f"content /tf: FAIL (missing {missing}; saw {sorted(frames)[:12]})"
            )
        else:
            lines.append(
                f"content /tf: PASS "
                f"(odom/base_link/lidar_link; {len(frames)} frames)"
            )
    else:
        lines.append(f"content /tf: PASS ({len(frames)} frames)")

    if robot in ("stretch", "stretch_wheeled", "reachy"):
        imu = data.get("imu")
        if not imu:
            lines.append("content /imu: FAIL (no message)")
        else:
            ax, ay, az = imu["a"]
            wx, wy, wz = imu["w"]
            g = math.sqrt(ax * ax + ay * ay + az * az)
            rate = math.sqrt(wx * wx + wy * wy + wz * wz)
            frame = str(imu.get("frame") or "")
            imu_ok_frame = (
                "imu" in frame.lower()
                or frame.endswith("base_imu")
                or frame.endswith("imu_link")
            )
            ok = 8.0 <= g <= 11.5 and rate < 0.5 and imu_ok_frame
            lines.append(
                f"content /imu: {'PASS' if ok else 'FAIL'} "
                f"(frame={frame}, |a|={g:.2f}, |w|={rate:.3f})"
            )

        beams = int(data.get("scan_beams") or 0)
        hits = int(data.get("scan_hits") or 0)
        sframe = str(data.get("scan_frame") or "")
        if beams < 100:
            lines.append(f"content /scan: FAIL (only {beams} beams)")
        else:
            lines.append(
                f"content /scan: PASS (beams={beams}, hits={hits}, frame={sframe})"
            )

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Lab robot sensor smoke (Phase 0)")
    ap.add_argument(
        "--robot",
        default="stretch",
        choices=["stretch", "stretch_wheeled", "reachy"],
    )
    ap.add_argument("--world", default="empty_world")
    ap.add_argument("--config", default=None, help="Scenario yaml basename or path")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--summary", default="/tmp/lab_robot_sensor_summary.txt")
    args = ap.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HUNAV_ISAAC_PROFILE", "debug")
    os.environ.setdefault("HUNAV_LAB_SENSORS", "1")
    # Cameras off by default for laptop VRAM.
    os.environ.setdefault("HUNAV_LAB_CAMERAS", "0")

    repo = _repo_root()
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from hunav_isaac_wrapper.sim_app_config import apply_profile_to_environ

    apply_profile_to_environ(profile="debug", headless=True)

    config = args.config
    if not config:
        if args.world == "museum":
            config = "museum_agents"
        elif args.world == "hospital":
            config = "hospital_agents"
        else:
            config = "empty_world_agents"

    # Resolve scenario path like other tools.
    scen = config
    if not os.path.isabs(scen) and not scen.endswith(".yaml"):
        candidate = os.path.join(repo, "src", "scenarios", f"{scen}.yaml")
        if os.path.isfile(candidate):
            scen = candidate
        else:
            scen = os.path.join(repo, "src", "scenarios", scen)

    print(
        f"[lab_robot_smoke] robot={args.robot} world={args.world} "
        f"config={scen} seconds={args.seconds}",
        flush=True,
    )

    from hunav_isaac_wrapper.teleop_hunav_sim import TeleopHuNavSim, simulation_app
    import omni
    import rclpy
    from rclpy.node import Node

    if not rclpy.ok():
        rclpy.init()

    node = TeleopHuNavSim(
        map_name=args.world,
        hunav_config=scen,
        robot_name=args.robot,
    )
    node.world.reset()
    if getattr(node, "_static_drive", False) and hasattr(node.robot, "try_init_articulation"):
        node.robot.try_init_articulation()

    _physx = getattr(omni.physx, "get_physx_interface", None) or getattr(
        omni.physx, "acquire_physx_interface", None
    )
    node.physx_interface = _physx()
    node.physx_sub = node.physx_interface.subscribe_physics_step_events(
        node._on_physics_step
    )

    class _Probe(Node):
        pass

    probe = _Probe("lab_robot_sensor_probe")
    capture_path = os.path.join(
        os.path.dirname(args.summary) or "/tmp",
        f"lab_robot_{args.robot}_content.json",
    )
    if os.path.isfile(capture_path):
        try:
            os.remove(capture_path)
        except OSError:
            pass
    capture_started = False
    capture_proc = None
    deadline = time.time() + float(args.seconds)
    topics_seen: Set[str] = set()
    steps = 0
    while simulation_app.is_running() and time.time() < deadline:
        node.world.step(render=True)
        steps += 1
        if getattr(node, "_lab_sensor_handles", None):
            from hunav_isaac_wrapper.lab_robot_sensors import tick_lab_sensor_handles

            tick_lab_sensor_handles(
                node._lab_sensor_handles,
                sim_time=float(node.world.current_time),
            )
        if getattr(node, "_static_drive", False) and hasattr(node.robot, "hold_joints"):
            node.robot.hold_joints()
        elif getattr(node, "_chassis_drive", False):
            node.robot.apply_cmd_vel(0.0, 0.0, node.world.get_physics_dt())
        # Start external content capture once topics are advertising.
        if not capture_started and steps >= 60:
            print(
                f"[lab_robot_smoke] starting external content capture → {capture_path}",
                flush=True,
            )
            capture_started = True
            # Run capture in background thread so sim keeps stepping.
            import threading

            def _run_cap():
                nonlocal capture_proc
                capture_proc = _external_content_capture(
                    args.robot, capture_path, wait_s=min(20.0, float(args.seconds) * 0.5)
                )

            threading.Thread(target=_run_cap, daemon=True).start()
        try:
            rclpy.spin_once(probe, timeout_sec=0.0)
        except Exception:
            pass
        if steps % 20 == 0:
            try:
                names = [n for n, _ in probe.get_topic_names_and_types()]
                topics_seen.update(names)
            except Exception:
                pass

    # Wait briefly for external capture thread to finish.
    wait_cap_until = time.time() + 25.0
    while time.time() < wait_cap_until and not os.path.isfile(capture_path):
        time.sleep(0.2)

    flat = sorted(topics_seen)
    want = {
        "stretch": ["/clock", "/tf", "/joint_states", "/scan", "/imu", "/odom"],
        "stretch_wheeled": ["/clock", "/tf", "/joint_states", "/scan", "/imu"],
        "reachy": ["/clock", "/tf", "/joint_states", "/scan", "/imu", "/odom"],
    }.get(args.robot, ["/clock"])
    if args.robot == "reachy" and os.environ.get("HUNAV_LAB_CAMERAS", "0") in (
        "1",
        "true",
        "yes",
        "on",
    ):
        want = list(want) + [
            "/left_camera/image_raw",
            "/right_camera/image_raw",
            "/left_camera/camera_info",
            "/right_camera/camera_info",
        ]

    lines = [
        f"robot={args.robot}",
        f"world={args.world}",
        f"steps={steps}",
        f"topics_total={len(flat)}",
    ]
    ok_topics = True
    for t in want:
        hit = any(t == x or x.endswith(t) for x in flat)
        ok_topics = ok_topics and hit
        lines.append(f"need {t}: {'PASS' if hit else 'MISS'}")

    captured = None
    if os.path.isfile(capture_path):
        import json

        try:
            with open(capture_path, "r", encoding="utf-8") as f:
                captured = json.load(f)
        except Exception:
            captured = capture_proc if isinstance(capture_proc, dict) else None
    elif isinstance(capture_proc, dict):
        captured = capture_proc

    content_lines = _evaluate_captured(args.robot, captured)
    if args.robot == "reachy" and getattr(node, "_chassis_drive", False):
        try:
            from hunav_isaac_wrapper.lab_robot_sensors import tick_lab_sensor_handles

            pos0, _ = node.robot.get_world_pose()
            dt = float(node.world.get_physics_dt())
            n_move = max(20, int(2.0 / max(dt, 1e-3)))
            for _ in range(n_move):
                node.robot.apply_cmd_vel(0.30, 0.0, dt)
                if getattr(node, "_lab_sensor_handles", None):
                    tick_lab_sensor_handles(
                        node._lab_sensor_handles,
                        sim_time=float(node.world.current_time),
                    )
                node.world.step(render=True)
            pos1, _ = node.robot.get_world_pose()
            moved = math.hypot(
                float(pos1[0]) - float(pos0[0]),
                float(pos1[1]) - float(pos0[1]),
            )
            ok_move = moved >= 0.25
            content_lines.append(
                f"content chassis_xy: {'PASS' if ok_move else 'FAIL'} "
                f"(Δxy={moved:.3f} m in ~2 s @ 0.30 m/s)"
            )
        except Exception as exc:
            content_lines.append(f"content chassis_xy: FAIL ({exc})")
    lines.append("--- content ---")
    lines.extend(content_lines)
    ok_content = all(": PASS" in ln for ln in content_lines) and bool(content_lines)
    ok_all = ok_topics and ok_content
    lines.append("overall: " + ("PASS" if ok_all else "FAIL"))
    lines.append("--- topics ---")
    lines.extend(flat)

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)
    print(f"[lab_robot_smoke] wrote {args.summary}", flush=True)

    try:
        node.hunav.close_hunav_nodes()
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass
    # Prefer explicit exit code (Kit close can swallow returns).
    os._exit(0 if ok_all else 2)


if __name__ == "__main__":
    raise SystemExit(main())

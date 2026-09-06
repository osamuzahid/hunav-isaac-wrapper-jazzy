#!/usr/bin/env python3
"""
Launch file for HuNav Isaac Wrapper simulation.
Launches the main script using the ROS2 launcher, which handles Isaac Sim python detection.
"""

# ---------------------------------------------------------------------------
# ORIGINALLY (upstream v2.0): two ExecuteProcess actions with IfCondition /
# UnlessCondition on `scenario`, always passing extra_args positionally; no
# SimulationApp profile launch arg. Kept conceptually below as comments in
# generate_launch_description history — replaced because empty `profile:=`
# must not inject a bare `--profile` into argv, and we need profile:=debug|laptop.
# ---------------------------------------------------------------------------
# launcher_with_scenario = ExecuteProcess(
#     cmd=['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher',
#          '--config', scenario, '--batch', extra_args],
#     condition=IfCondition(scenario), ...)
# launcher_interactive = ExecuteProcess(
#     cmd=['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher', extra_args],
#     condition=UnlessCondition(scenario), ...)
# ---------------------------------------------------------------------------
# PATCH (isaac-social-nav): OpaqueFunction builds cmd lists; optional profile
# arg; only append --profile when non-empty. Important for laptop/debug Kit.
# ---------------------------------------------------------------------------

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition


def _launch_setup(context, *args, **kwargs):
    """Build launcher cmds; only forward --profile when launch arg is non-empty."""
    scenario = LaunchConfiguration('scenario').perform(context)
    profile = LaunchConfiguration('profile').perform(context).strip()
    extra_args = LaunchConfiguration('extra_args').perform(context).strip()

    profile_args = ['--profile', profile] if profile else []
    extra = extra_args.split() if extra_args else []

    actions = []

    if scenario:
        cmd = (
            ['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher',
             '--config', scenario, '--batch']
            + profile_args
            + extra
        )
        actions.append(
            ExecuteProcess(
                cmd=cmd,
                output='screen',
                name='hunav_isaac_launcher_scenario',
            )
        )
    else:
        cmd = (
            ['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher']
            + profile_args
            + extra
        )
        actions.append(
            ExecuteProcess(
                cmd=cmd,
                output='screen',
                name='hunav_isaac_launcher_interactive',
            )
        )

    return actions


def generate_launch_description():
    """Generate launch description that uses the ROS2 launcher."""

    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='',
        description='Scenario configuration file to use (optional - will use interactive mode if not specified)'
    )

    batch_arg = DeclareLaunchArgument(
        'batch',
        default_value='false',
        description='Run in batch mode (non-interactive)'
    )

    extra_args_arg = DeclareLaunchArgument(
        'extra_args',
        default_value='',
        description='Additional arguments to pass to the main script'
    )

    profile_arg = DeclareLaunchArgument(
        'profile',
        default_value='',
        description=(
            'SimulationApp profile: default|lab (1280x720 windowed) or '
            'debug|laptop (960x540 headless). Empty = env HUNAV_ISAAC_PROFILE or default. '
            'Example: profile:=debug'
        ),
    )

    return LaunchDescription([
        scenario_arg,
        batch_arg,
        extra_args_arg,
        profile_arg,
        LogInfo(msg=['Launching HuNav Isaac Wrapper...']),
        LogInfo(msg=['Use scenario:=file.yaml and/or profile:=debug|laptop|default|lab']),
        LogInfo(msg=['Or set HUNAV_ISAAC_PROFILE; empty profile uses env/default']),
        OpaqueFunction(function=_launch_setup),
    ])

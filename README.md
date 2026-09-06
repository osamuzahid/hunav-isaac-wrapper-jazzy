# HuNav Isaac Wrapper (Isaac Sim 6 / ROS 2 Jazzy)

Isaac Sim wrapper for [HuNavSim](https://github.com/robotics-upo/hunav_sim): SimulationApp lifecycle, USD world loading by descriptor or URI, HuNav pedestrian visualisation and animation, ROS 2 bridge (clock, state, `/cmd_vel`), and reusable lidar / camera / IMU construction.

This tree is a ROS 2 **Jazzy** and **NVIDIA Isaac Sim 6.0.1** port of [robotics-upo/Hunav_isaac_wrapper](https://github.com/robotics-upo/Hunav_isaac_wrapper) `v2.0`. Provenance: [UPSTREAM.md](UPSTREAM.md).

Campaign worlds, Stretch/Reachy meshes, crowd YAML, and hop launchers are not in this repository. Lab robots are discovered from `config/robots/<name>/robot.yaml` when that tree is supplied separately.

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- NVIDIA Isaac Sim 6.0.1 (workstation install; do not redistribute Isaac Sim from this repo)
- [HuNavSim](https://github.com/robotics-upo/hunav_sim) on Jazzy (`hunav_msgs` and related packages)

## Build

Place this package under a colcon workspace `src/` directory (this repository *is* the package root; the ROS package lives in `src/`).

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-geometry-msgs ros-jazzy-nav-msgs ros-jazzy-sensor-msgs ros-jazzy-tf2-ros
pip install pyyaml numpy matplotlib
colcon build --packages-select hunav_isaac_wrapper
source install/setup.bash
```

`setup_workspace.sh` (run from `src/`) prints the same Jazzy source line if `ROS_DISTRO` is unset.

## empty_world smoke

`src/worlds/empty_world.usd` is the generic load/shutdown world. With Isaac Sim 6.0.1 available as `~/isaacsim/python.sh`:

```bash
./launch_hunav_isaac.sh --config empty_world_agents --world empty_world --robot carter --batch --debug
```

Equivalent:

```bash
bash ~/isaacsim/python.sh src/scripts/main.py \
  --config empty_world_agents --world empty_world --robot carter --batch --debug
```

`--debug` / `HUNAV_ISAAC_PROFILE=debug` uses a 960×540 headless SimulationApp profile. Stock HuNav `hospital` and `office` USD, maps, and agent YAML from upstream `v2.0` remain for the original demos.

Upstream CDN robots: `jetbot`, `create3`, `carter`, `carter_ROS`.

## Tests (no GPU)

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

These cover occupancy-grid math and `robot.yaml` catalog path logic. They do not start Isaac Sim.

## Licence

MIT. Copyright Miguel Escudero Jiménez (see `src/package.xml`). Root [LICENSE](LICENSE) is the standard MIT text; there was no root licence file on upstream `v2.0`. This is a private attributed derivative — see [UPSTREAM.md](UPSTREAM.md).

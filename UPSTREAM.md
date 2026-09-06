# Upstream

- Original URL: https://github.com/robotics-upo/Hunav_isaac_wrapper
- Base SHA: `eaaba2dd0d1559793159063a72e1fe34e0b75299` (robotics-upo `v2.0`)
- Freeze SHA compared: `64e8bb3c69eb7cd467e87e6a4254d7471858698a`
- Licence: **MIT** declared in `src/package.xml` (`<license>MIT</license>`) and `src/setup.py` (`license='MIT'`). Upstream `v2.0` has **no root LICENSE file**.
- Date inspected: 2026-09-06

Copyright / author preserved from upstream package metadata: Miguel Escudero Jiménez `<mescjim@upo.es>` (`package.xml`, `setup.py`, `src/hunav_isaac_wrapper/__init__.py`).

This repository is a private attributed derivative. There is no written robotics-upo grant for republication.

## Divergence summary

Relative to `eaaba2d`, this tree ports the wrapper to **Isaac Sim 6.0.1** and **ROS 2 Jazzy**, and adds configuration-driven robot/world loading plus reusable lab sensor interfaces.

Not in this repository (handover assets / platform trees):

- CUCR converted worlds and `src/worlds/assets/` payloads
- Stretch and Reachy USD/URDF/meshes
- Campaign crowd YAML, hop launchers, and lab `robot.yaml` descriptors

Kept from upstream `v2.0`: stock HuNav `hospital` / `office` USD, maps, and agent YAML; CDN Carter zip (`src/config/robots/nova_carter_ros2_sensors.zip`); `carter_navigation_params.yaml`; `hunav_isaac.rviz`; generic behaviour-tree XML (including `warehouse_agents__agent_*.xml`). The upstream warehouse *world* USD/map/scenario files are omitted; `empty_world.usd` is the generic load/shutdown world.

`nova_carter_ros2_sensors.zip` is a vendored upstream binary; no nested licence file was inspected at freeze.

Isaac Sim, Kit, and People CDN assets are runtime-only and are not redistributed here.

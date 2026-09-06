"""Occupancy-grid math (no Isaac Sim, no GPU)."""

from __future__ import annotations

import numpy as np

from hunav_isaac_wrapper.occupancy_path import OccupancyMap, densify_waypoints


def _open_map(height: int = 20, width: int = 20, resolution: float = 0.05) -> OccupancyMap:
    img = np.full((height, width), 255, dtype=np.uint8)
    om = OccupancyMap(
        image=img,
        resolution=resolution,
        origin_x=-1.0,
        origin_y=-1.0,
    )
    om.rebuild_navigable()
    return om


def test_world_grid_roundtrip_cell_centres() -> None:
    om = _open_map()
    for col in (0, 3, om.width - 1):
        for row in (0, 5, om.height - 1):
            x, y = om.grid_to_world(col, row)
            got = om.world_to_grid(x, y)
            assert got == (col, row)


def test_world_to_grid_uses_floor_not_round() -> None:
    om = _open_map()
    # Cell (0, height-1) covers x,y in [origin, origin+res).
    col, row = om.world_to_grid(-1.0, -1.0)
    assert col == 0
    assert row == om.height - 1
    col2, row2 = om.world_to_grid(-1.0 + 0.049, -1.0 + 0.049)
    assert (col2, row2) == (col, row)


def test_open_map_is_navigable() -> None:
    om = _open_map()
    assert om.is_navigable_world(*om.grid_to_world(1, 1))


def test_occupied_cell_not_navigable() -> None:
    om = _open_map()
    om.image[10, 4] = 0
    om.inflation_radius_m = 0.0
    om.rebuild_navigable()
    x, y = om.grid_to_world(4, 10)
    assert not om.is_navigable_world(x, y)


def test_densify_waypoints_keeps_endpoints() -> None:
    path = densify_waypoints([(0.0, 0.0), (3.0, 0.0)], spacing_m=1.0)
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (3.0, 0.0)
    assert len(path) >= 4
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        assert (dx * dx + dy * dy) ** 0.5 <= 1.0 + 1e-6

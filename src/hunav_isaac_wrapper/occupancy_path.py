"""
Occupancy-grid path planning for HuNav goal chains.

PATCH (isaac-social-nav): HuNav SFM walks straight chords between goals and has
no global planner. Plan on a ROS map_server YAML + image (e.g. maps/museum.yaml)
and densify the path into waypoints so agents can circulate a whole building.

Swap: pass a world's occupancy YAML to plan_*_routes.py — this module is not
robot-specific. Robot / planner stay --robot and Nav2-or-ESC.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

WorldPoint = Tuple[float, float]
GridPoint = Tuple[int, int]  # (col, row)


def _load_gray_image(path: str) -> np.ndarray:
    """Load a single-channel occupancy image as uint8 (0=black, 255=white)."""
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except ImportError:
        import matplotlib.pyplot as plt

        img = plt.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)
        if img.dtype != np.uint8:
            if float(np.nanmax(img)) <= 1.0 + 1e-6:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img


@dataclass
class OccupancyMap:
    """ROS-style occupancy grid used for A* pedestrian routing."""

    image: np.ndarray  # row-major, row 0 = top of image (ROS map convention)
    resolution: float
    origin_x: float
    origin_y: float
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    negate: int = 0
    inflation_radius_m: float = 0.35
    free_pixel_min: int = 250  # strict free for sparse museum free paint

    navigable: Optional[np.ndarray] = None

    @classmethod
    def from_yaml(cls, yaml_path: str, inflation_radius_m: float = 0.35) -> "OccupancyMap":
        yaml_path = os.path.abspath(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        image_name = meta["image"]
        image_path = image_name if os.path.isabs(image_name) else os.path.join(
            os.path.dirname(yaml_path), image_name
        )
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        om = cls(
            image=_load_gray_image(image_path),
            resolution=float(meta["resolution"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            occupied_thresh=float(meta.get("occupied_thresh", 0.65)),
            free_thresh=float(meta.get("free_thresh", 0.196)),
            negate=int(meta.get("negate", 0)),
            inflation_radius_m=float(inflation_radius_m),
        )
        om.rebuild_navigable()
        return om

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def rebuild_navigable(self) -> None:
        """Mark free cells after inflating occupied (and optionally unknown)."""
        pix = self.image.astype(np.float64)
        if self.negate:
            occ_prob = pix / 255.0
        else:
            occ_prob = (255.0 - pix) / 255.0

        occupied = occ_prob > self.occupied_thresh
        # Prefer explicit free paint when the map is mostly "unknown" gray
        # (museum.png: walls dark, free white≈254, unknown≈205).
        free = (pix >= self.free_pixel_min) & ~occupied

        rad = max(0, int(math.ceil(self.inflation_radius_m / self.resolution)))
        if rad > 0 and occupied.any():
            inflated = _dilate_bool(occupied, rad)
            free = free & ~inflated
        self.navigable = free

    def world_to_grid(self, x: float, y: float) -> GridPoint:
        # Floor, not round: cell i covers [origin + i*res, origin + (i+1)*res).
        # round() sends a cell-centre at *.5 into the next cell (Python ties-to-even),
        # so A* centres on odd columns looked occupied after densify.
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row_from_bottom = int(math.floor((y - self.origin_y) / self.resolution))
        row = self.height - 1 - row_from_bottom
        return col, row

    def grid_to_world(self, col: int, row: int) -> WorldPoint:
        row_from_bottom = self.height - 1 - row
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row_from_bottom + 0.5) * self.resolution
        return x, y

    def is_navigable_world(self, x: float, y: float) -> bool:
        col, row = self.world_to_grid(x, y)
        return self._in_bounds(col, row) and bool(self.navigable[row, col])

    def nearest_navigable(self, x: float, y: float, max_radius_m: float = 2.0) -> Optional[WorldPoint]:
        """Snap a world point onto the nearest navigable cell (spiral search)."""
        if self.is_navigable_world(x, y):
            return x, y
        col0, row0 = self.world_to_grid(x, y)
        max_r = max(1, int(math.ceil(max_radius_m / self.resolution)))
        best = None
        best_d2 = None
        for r in range(1, max_r + 1):
            for dc in range(-r, r + 1):
                for dr in (-r, r):
                    cand = (col0 + dc, row0 + dr)
                    d2 = self._try_nav_cand(cand, col0, row0)
                    if d2 is not None and (best_d2 is None or d2 < best_d2):
                        best_d2 = d2
                        best = cand
            for dr in range(-r + 1, r):
                for dc in (-r, r):
                    cand = (col0 + dc, row0 + dr)
                    d2 = self._try_nav_cand(cand, col0, row0)
                    if d2 is not None and (best_d2 is None or d2 < best_d2):
                        best_d2 = d2
                        best = cand
            if best is not None:
                return self.grid_to_world(*best)
        return None

    def _try_nav_cand(self, cand: GridPoint, col0: int, row0: int) -> Optional[float]:
        c, r = cand
        if not self._in_bounds(c, r) or not self.navigable[r, c]:
            return None
        return float((c - col0) ** 2 + (r - row0) ** 2)

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def plan(
        self,
        start: WorldPoint,
        goal: WorldPoint,
        waypoint_spacing_m: float = 1.0,
    ) -> Optional[List[WorldPoint]]:
        """A* from start to goal; return densified world waypoints (incl. endpoints)."""
        s = self.nearest_navigable(*start)
        g = self.nearest_navigable(*goal)
        if s is None or g is None:
            return None
        sc, sr = self.world_to_grid(*s)
        gc, gr = self.world_to_grid(*g)
        grid_path = self._astar((sc, sr), (gc, gr))
        if not grid_path:
            return None
        world = [self.grid_to_world(c, r) for c, r in grid_path]
        dense = densify_waypoints(world, spacing_m=waypoint_spacing_m)
        return self._snap_path_to_navigable(dense)

    def plan_route(
        self,
        waypoints: Sequence[WorldPoint],
        waypoint_spacing_m: float = 1.0,
    ) -> Optional[List[WorldPoint]]:
        """Plan through a sequence of key poses; concatenate densified segments."""
        if len(waypoints) < 2:
            return list(waypoints) if waypoints else None
        out: List[WorldPoint] = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            seg = self.plan(a, b, waypoint_spacing_m=waypoint_spacing_m)
            if seg is None:
                return None
            if out:
                out.extend(seg[1:])  # drop duplicate joint
            else:
                out.extend(seg)
        return self._snap_path_to_navigable(out)

    def _snap_path_to_navigable(
        self, path: Sequence[WorldPoint], max_radius_m: float = 0.6
    ) -> Optional[List[WorldPoint]]:
        """Keep densified waypoints on inflated-free cells.

        A* visits navigable cell centres, but 1 m samples along that polyline
        can round onto a neighbouring inflated cell (thin corridor / corner).
        Snap those hits back; fail the plan if a sample cannot be rescued.
        """
        if not path:
            return []
        out: List[WorldPoint] = []
        for x, y in path:
            if self.is_navigable_world(x, y):
                pt = (float(x), float(y))
            else:
                snapped = self.nearest_navigable(x, y, max_radius_m=max_radius_m)
                if snapped is None:
                    return None
                pt = (float(snapped[0]), float(snapped[1]))
            if out and math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) < 0.05:
                out[-1] = pt
            else:
                out.append(pt)
        return out

    def _astar(self, start: GridPoint, goal: GridPoint) -> Optional[List[GridPoint]]:
        if not self._in_bounds(*start) or not self._in_bounds(*goal):
            return None
        if not self.navigable[start[1], start[0]] or not self.navigable[goal[1], goal[0]]:
            return None

        def h(c: int, r: int) -> float:
            return math.hypot(c - goal[0], r - goal[1])

        open_heap: List[Tuple[float, float, GridPoint]] = []
        heapq.heappush(open_heap, (h(*start), 0.0, start))
        came: dict = {start: None}
        gscore = {start: 0.0}
        closed = set()
        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2)),
        )

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = []
                cur = current
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                path.reverse()
                return path
            closed.add(current)
            c0, r0 = current
            for dc, dr, step in neighbors:
                nc, nr = c0 + dc, r0 + dr
                if not self._in_bounds(nc, nr) or not self.navigable[nr, nc]:
                    continue
                # Corner cut: both orthogonal neighbors must be free for diagonals
                if dc != 0 and dr != 0:
                    if not self.navigable[r0, nc] or not self.navigable[nr, c0]:
                        continue
                ng = g + step
                nxt = (nc, nr)
                if ng + 1e-9 < gscore.get(nxt, float("inf")):
                    gscore[nxt] = ng
                    came[nxt] = current
                    heapq.heappush(open_heap, (ng + h(nc, nr), ng, nxt))
        return None


def densify_waypoints(
    path: Sequence[WorldPoint], spacing_m: float = 1.0
) -> List[WorldPoint]:
    """Keep endpoints; insert points so consecutive spacing ≈ spacing_m."""
    if not path:
        return []
    if len(path) == 1 or spacing_m <= 0:
        return [(float(x), float(y)) for x, y in path]

    out: List[WorldPoint] = [(float(path[0][0]), float(path[0][1]))]
    traveled = 0.0
    next_emit = spacing_m
    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        t0 = traveled
        t1 = traveled + seg
        while next_emit < t1 - 1e-9:
            u = (next_emit - t0) / seg
            out.append((float(x0 + u * (x1 - x0)), float(y0 + u * (y1 - y0))))
            next_emit += spacing_m
        traveled = t1
    end = (float(path[-1][0]), float(path[-1][1]))
    if math.hypot(end[0] - out[-1][0], end[1] - out[-1][1]) > 0.05:
        out.append(end)
    else:
        out[-1] = end
    return out


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation without scipy (square kernel of given radius)."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        out[y0:y1, x0:x1] = True
    return out


def resolve_map_yaml(
    map_name_or_file: str, maps_dir: str
) -> str:
    """Resolve 'museum' or 'museum.yaml' to an absolute maps/*.yaml path."""
    name = map_name_or_file.strip()
    if not name.endswith(".yaml") and not name.endswith(".yml"):
        name = name + ".yaml"
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    candidate = os.path.join(maps_dir, name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Occupancy map YAML not found: {candidate}")

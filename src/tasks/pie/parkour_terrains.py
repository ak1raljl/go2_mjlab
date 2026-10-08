"""Route terrains inspired by Extreme Parkour and the PIE terrain families.

Geometry and ordered support points are generated together. mjlab's named
patch transport applies the same tile/world transform to these explicit
points; they are not randomly sampled flat patches.
"""

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

from .roughness import roughen_box_surfaces


TerrainKind = Literal[
  "flat", "hurdle", "step", "gap", "platform", "stairs_up", "stairs_down",
  "slope_up", "slope_down",
]
ROUTE_POINTS = 8


def _place_obstacles(widths, start, end, min_gap, rng):
  """Randomize every free interval while retaining approach/landing clearance."""
  widths = np.asarray(widths, dtype=float)
  slack = end - start - widths.sum() - min_gap * (len(widths) - 1)
  if slack < 0:
    raise ValueError("Obstacles do not fit with the requested minimum clearance.")
  gaps = slack * rng.dirichlet(np.ones(len(widths) + 1))
  gaps[1:-1] += min_gap
  left = start + gaps[0]
  intervals = []
  for index, width in enumerate(widths):
    intervals.append((left, left + width))
    left += width + gaps[index + 1]
  return intervals


@dataclass(kw_only=True)
class ParkourTerrainCfg(SubTerrainCfg):
  kind: TerrainKind = "flat"
  lateral_range: float = 0.35
  hurdle_height_range: tuple[float, float] = (0.05, 0.75)
  platform_height_range: tuple[float, float] = (0.05, 0.75)
  step_height_range: tuple[float, float] = (0.03, 0.35)
  stair_height_range: tuple[float, float] = (0.03, 0.25)
  gap_width_range: tuple[float, float] = (0.10, 1.0)
  obstacle_min_spacing: float = 1.1
  """Minimum clear ground between consecutive hurdles/gaps, in metres."""
  platform_count_range: tuple[int, int] = (1, 3)
  platform_length_range: tuple[float, float] = (1.4, 2.2)
  platform_min_spacing: float = 1.6
  slope_angle_range: tuple[float, float] = (5.0, 25.0)
  """Slope inclination in degrees, interpolated by difficulty."""
  slope_length_range: tuple[float, float] = (4.0, 6.0)
  # Reference Extreme Parkour roughness; stairs explicitly remain smooth.
  roughness_height_range: tuple[float, float] = (0.02, 0.06)
  roughness_horizontal_scale: float = 0.05
  roughness_downsampled_scale: float = 0.075
  roughness_vertical_scale: float = 0.005
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] = field(default_factory=lambda: {
    "route_goals": FlatPatchSamplingCfg(num_patches=ROUTE_POINTS, patch_radius=0.1),
  })

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    length, width = self.size
    if length < 16.0 or width < 3.0:
      raise ValueError("Parkour routes require tiles at least 16 x 3 metres.")
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    body = spec.body("terrain")
    geometries = []
    mid = width / 2

    def height(bounds):
      return bounds[0] + difficulty * (bounds[1] - bounds[0])

    def box(x0, x1, y0, y1, top, bottom=-0.15):
      if x1 <= x0 or y1 <= y0 or top <= bottom:
        raise ValueError("Invalid parkour box dimensions.")
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x0 + x1) / 2, (y0 + y1) / 2, (top + bottom) / 2),
        size=((x1 - x0) / 2, (y1 - y0) / 2, (top - bottom) / 2),
        rgba=(0.45, 0.50, 0.55, 1.0),
      )
      geometries.append(TerrainGeometry(geom=geom))

    origin = np.array([1.0, mid, 0.0])
    goals = np.zeros((ROUTE_POINTS, 3))
    goals[:, 0] = np.linspace(2.0, length - 1.0, ROUTE_POINTS)
    goals[:, 1] = mid + rng.uniform(-self.lateral_range, self.lateral_range, ROUTE_POINTS)
    goals[[0, -1], 1] = mid
    height_profile = None
    if self.kind != "gap":
      box(0, length, 0, width, 0)

    if self.kind == "hurdle":
      depth = 0.1 + 0.3 * difficulty
      intervals = _place_obstacles(np.full(6, depth), 3.0, length - 2.0,
                                   self.obstacle_min_spacing, rng)
      for i, (left, right) in enumerate(intervals):
        h = height(self.hurdle_height_range) * rng.uniform(0.8, 1.0)
        y = goals[i + 1, 1]
        box(left, right, y - 0.95, y + 0.95, h)
        goals[i + 1, 0] = right + 0.65
    elif self.kind == "gap":
      gap_width = height(self.gap_width_range)
      pit_depth = rng.uniform(0.5, 1.0)
      box(0, length, 0, width, -pit_depth, -pit_depth - 0.15)
      left = 0.0
      intervals = _place_obstacles(np.full(6, gap_width), 3.0, length - 2.0,
                                   self.obstacle_min_spacing, rng)
      for i, (gap_start, gap_end) in enumerate(intervals):
        box(left, gap_start, 0, width, 0)
        left = gap_end
        # Land-side waypoints, with enough support for the robot footprint.
        goals[i + 1, 0] = left + 0.55
      box(left, length, 0, width, 0)
    elif self.kind == "step":
      start, end = rng.uniform(2.5, 3.5), length - rng.uniform(2.5, 3.5)
      lengths = 1.0 + (end - start - 6.0) * rng.dirichlet(np.ones(6) * 2)
      boundaries = np.r_[start, start + np.cumsum(lengths)]
      heights = np.array([1, 2, 3, 3, 2, 1]) * height(self.step_height_range)
      for i, top in enumerate(heights):
        y = goals[i + 1, 1]
        box(boundaries[i], boundaries[i + 1], y - 1.0, y + 1.0, top)
        goals[i + 1, 0] = (boundaries[i] + boundaries[i + 1]) / 2
        goals[i + 1, 2] = top
    elif self.kind == "platform":
      low, high = self.platform_count_range
      if not 1 <= low <= high <= 3:
        raise ValueError("platform_count_range must lie in [1, 3] for eight route points.")
      count = rng.integers(low, high + 1)
      lengths = rng.uniform(*self.platform_length_range, count)
      intervals = _place_obstacles(lengths, 2.75, length - 2.75,
                                   self.platform_min_spacing, rng)
      route = [[2.0, mid, 0.0], [length - 1.0, mid, 0.0]]
      floor_intervals = []
      previous_landing = 2.0
      for left, right in intervals:
        y = mid + rng.uniform(-self.lateral_range, self.lateral_range)
        top = height(self.platform_height_range) * rng.uniform(0.85, 1.0)
        box(left, right, y - 1.1, y + 1.1, top)
        route.extend([[(left + right) / 2, y, top], [right + 0.65, y, 0]])
        floor_intervals.append((previous_landing, left - 0.65))
        previous_landing = right + 0.65
      floor_intervals.append((previous_landing, length - 1.0))
      # Fill shorter routes with genuine ground waypoints, never duplicate
      # padded goals or place interpolated points on a platform's vertical face.
      while len(route) < ROUTE_POINTS:
        index = max(range(len(floor_intervals)),
                    key=lambda i: floor_intervals[i][1] - floor_intervals[i][0])
        left, right = floor_intervals.pop(index)
        x = (left + right) / 2
        route.append([x, mid, 0])
        floor_intervals.extend(((left, x), (x, right)))
      goals[:] = sorted(route, key=lambda p: p[0])
    elif self.kind in ("stairs_up", "stairs_down"):
      rise = height(self.stair_height_range)
      count, tread = 8, 0.30
      start = length / 2 - count * tread / 2 + rng.uniform(-1.0, 1.0)
      top = count * rise
      if self.kind == "stairs_down":
        box(0, start, 0, width, top)
        origin[2] = top
      for i in range(count):
        z = (i + 1 if self.kind == "stairs_up" else count - i - 1) * rise
        if z > 0:
          box(start + i * tread, start + (i + 1) * tread, 0, width, z)
      if self.kind == "stairs_up":
        box(start + count * tread, length, 0, width, top)
      goals[:, 1] = mid
      goals[:, 0] = [2, start - 0.6, start + 0.45, start + 1.05,
                     start + 1.65, start + 2.25, start + 3.0, length - 1]
      levels = np.clip(np.floor((goals[:, 0] - start) / tread) + 1, 0, count)
      goals[:, 2] = (levels if self.kind == "stairs_up" else count - levels) * rise
    elif self.kind in ("slope_up", "slope_down"):
      start = rng.uniform(3.0, 5.0)
      ramp_length = rng.uniform(*self.slope_length_range)
      grade = np.tan(np.deg2rad(height(self.slope_angle_range)))
      top = ramp_length * grade

      def height_profile(x, y):
        elevation = np.clip(x - start, 0.0, ramp_length) * grade
        return elevation if self.kind == "slope_up" else top - elevation

      goals[:, 1] = mid
      goals[:, 2] = height_profile(goals[:, 0], goals[:, 1])
      origin[2] = height_profile(origin[0], origin[1])
    elif self.kind != "flat":
      raise ValueError(f"Unknown parkour terrain: {self.kind}")

    if not np.all(np.diff(goals[:, 0]) > 0):
      raise ValueError("Route goals must advance strictly along the tile.")
    if self.kind not in ("stairs_up", "stairs_down"):
      geometries = roughen_box_surfaces(
        spec, geometries, origin, goals, self, rng, height_profile=height_profile,
      )
    return TerrainOutput(origin=origin, geometries=geometries, flat_patches={"route_goals": goals})


PIE_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(18.0, 4.0), border_width=5.0, num_rows=10, num_cols=20,
  curriculum=True, add_lights=True,
  sub_terrains={
    name: ParkourTerrainCfg(
      kind=name, proportion=weight,
      roughness_height_range=(
        (0.0, 0.0) if name.startswith("stairs_")
        else (0.04, 0.10) if name == "flat" else (0.02, 0.06)
      ),
    )
    for name, weight in {
      "flat": 0.10, "hurdle": 0.15, "step": 0.15, "gap": 0.15,
      "platform": 0.15, "stairs_up": 0.10, "stairs_down": 0.10,
      "slope_up": 0.05, "slope_down": 0.05,
    }.items()
  },
)

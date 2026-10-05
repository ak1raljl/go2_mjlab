"""AMP terrain mix matching amp_go2's proportions and difficulty ranges."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.interpolate import RectBivariateSpline

from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfDiscreteObstaclesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainOutput


@dataclass(kw_only=True)
class HfRoughPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
  """Add linearly interpolated uniform noise to a pyramid slope."""

  noise_range: tuple[float, float] = (-0.05, 0.05)
  noise_step: float = 0.005
  downsampled_scale: float = 0.2

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    output = super().function(difficulty, spec, rng)
    geometry = output.geometries[0]
    field, geom = geometry.hfield, geometry.geom
    assert field is not None and geom is not None

    # Recover signed physical heights before adding noise. Keeping the noise
    # on the center platform matches amp_go2's rough slope generator.
    heights = np.asarray(field.userdata).reshape(field.nrow, field.ncol)
    heights = heights * field.size[2] + geom.pos[2]
    coarse_shape = tuple(int(size / self.downsampled_scale) for size in self.size)
    noise_values = np.arange(
      self.noise_range[0], self.noise_range[1] + self.noise_step * 0.5, self.noise_step
    )
    coarse_noise = rng.choice(noise_values, size=coarse_shape)
    interpolate = RectBivariateSpline(
      np.linspace(0, self.size[0], coarse_shape[0]),
      np.linspace(0, self.size[1], coarse_shape[1]),
      coarse_noise,
      kx=1,
      ky=1,
    )
    noise = interpolate(
      np.linspace(0, self.size[0], field.nrow),
      np.linspace(0, self.size[1], field.ncol),
    )
    heights += np.rint(noise / self.vertical_scale) * self.vertical_scale

    minimum, maximum = float(heights.min()), float(heights.max())
    height_range = max(maximum - minimum, self.vertical_scale)
    normalized = (heights - minimum) / height_range
    field.userdata = normalized.astype(np.float32).ravel().tolist()
    field.size[2] = height_range
    field.size[3] = height_range * self.base_thickness_ratio
    geom.pos[2] = minimum
    geom.material = color_by_height(
      spec, heights / self.vertical_scale, field.name + "_rough", normalized
    )

    # amp_go2 uses the maximum height in the central 2 x 2 m as the spawn
    # origin. The ordinary reset then adds +/-1 m XY and the default base height.
    half_width = int(1.0 / self.horizontal_scale)
    cx, cy = field.nrow // 2, field.ncol // 2
    output.origin[2] = heights[
      cx - half_width : cx + half_width, cy - half_width : cy + half_width
    ].max()
    return output


def make_amp_rough_terrains_cfg(play: bool = False) -> TerrainGeneratorCfg:
  """Create fresh terrain objects without mutating mjlab's shared presets."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=5 if play else 10,
    num_cols=20,
    curriculum=not play,
    sub_terrains={
      "slope_inv": HfPyramidSlopedTerrainCfg(
        proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0, inverted=True,
      ),
      "slope": HfPyramidSlopedTerrainCfg(
        proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0,
      ),
      "rough_slope": HfRoughPyramidSlopedTerrainCfg(
        proportion=0.2, slope_range=(0.0, 0.4), platform_width=3.0,
      ),
      # These names follow amp_go2: stairs_up uses negative pyramid steps
      # because the robot starts on the center platform and walks outwards.
      "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.25, step_height_range=(0.05, 0.23),
        step_width=0.31, platform_width=3.0,
      ),
      "stairs_down": BoxPyramidStairsTerrainCfg(
        proportion=0.25, step_height_range=(0.05, 0.23),
        step_width=0.31, platform_width=3.0,
      ),
      "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
        proportion=0.2, obstacle_width_range=(1.0, 2.0),
        obstacle_height_range=(0.05, 0.25), num_obstacles=20, platform_width=3.0,
      ),
    },
    add_lights=True,
  )

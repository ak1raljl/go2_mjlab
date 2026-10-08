"""Add physical surface roughness while retaining sharp obstacle boundaries."""

import uuid

import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from mjlab.terrains.terrain_generator import TerrainGeometry


def _surface_height(x, y, x0, x1, y0, y1, elevation):
  """Interpolate MuJoCo's two triangles along the lower-left/upper-right diagonal."""
  rows, cols = elevation.shape
  u = np.clip((x - x0) / (x1 - x0) * (cols - 1), 0, cols - 1)
  v = np.clip((y - y0) / (y1 - y0) * (rows - 1), 0, rows - 1)
  col, row = min(int(u), cols - 2), min(int(v), rows - 2)
  u, v = u - col, v - row
  z00, z10 = elevation[row, col:col + 2]
  z01, z11 = elevation[row + 1, col:col + 2]
  if u >= v:
    return z00 * (1 - u) + z10 * (u - v) + z11 * v
  return z00 * (1 - v) + z01 * (v - u) + z11 * u


def roughen_box_surfaces(spec, geometries, origin, goals, cfg, rng, height_profile=None):
  """Replace each box by a solid heightfield covering exactly its footprint.

  Use a shared XY noise field across all surfaces, including elevated tops
  and pit bottoms. Separate patches preserve vertical walls and gap widths;
  sampling one heightfield over the whole tile would slope obstacle edges.
  No flat collision box remains to clip the negative half of the noise.
  """
  low, high = cfg.roughness_height_range
  if not 0 <= low <= high < 0.15:
    raise ValueError("roughness_height_range must satisfy 0 <= min <= max < 0.15 m.")
  if min(cfg.roughness_horizontal_scale, cfg.roughness_downsampled_scale,
         cfg.roughness_vertical_scale) <= 0:
    raise ValueError("Roughness grid scales must be positive.")
  if high == 0 and height_profile is None:
    return geometries
  amplitude = rng.uniform(low, high)
  length, width = cfg.size
  coarse_x = np.linspace(0, length, int(np.ceil(length / cfg.roughness_downsampled_scale)) + 1)
  coarse_y = np.linspace(0, width, int(np.ceil(width / cfg.roughness_downsampled_scale)) + 1)
  # Match the reference's quantized uniform heights before spatial interpolation.
  steps = int(amplitude / cfg.roughness_vertical_scale)
  coarse = rng.integers(-steps, steps + 1, size=(len(coarse_y), len(coarse_x)))
  coarse = coarse * cfg.roughness_vertical_scale
  noise = RegularGridInterpolator((coarse_y, coarse_x), coarse, bounds_error=False, fill_value=None)
  body = spec.body("terrain")
  rough_geometries, surfaces = [], []
  for item in geometries:
    geom = item.geom
    if geom is None or geom.type != mujoco.mjtGeom.mjGEOM_BOX:
      raise ValueError("Rough parkour generation expects axis-aligned boxes.")
    center, half = np.asarray(geom.pos), np.asarray(geom.size)
    x0, x1 = center[0] - half[0], center[0] + half[0]
    y0, y1 = center[1] - half[1], center[1] + half[1]
    top, bottom = center[2] + half[2], center[2] - half[2]
    xs = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / cfg.roughness_horizontal_scale)) + 1))
    ys = np.linspace(y0, y1, max(2, int(np.ceil((y1 - y0) / cfg.roughness_horizontal_scale)) + 1))
    xx, yy = np.meshgrid(xs, ys)
    nominal = top if height_profile is None else height_profile(xx, yy)
    elevation = nominal + noise(np.stack((yy, xx), axis=-1))
    zmin, zmax = float(elevation.min()), float(elevation.max())
    span = max(zmax - zmin, 1e-6)
    normalized = ((elevation - zmin) / span).astype(np.float32)
    field = spec.add_hfield(
      name=f"parkour_rough_{uuid.uuid4().hex}",
      size=(half[0], half[1], span, zmin - bottom),
      nrow=len(ys), ncol=len(xs), userdata=normalized.ravel().tolist(),
    )
    rough_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
      pos=(center[0], center[1], zmin), rgba=geom.rgba,
    )
    spec.delete(geom)
    rough_geometries.append(TerrainGeometry(geom=rough_geom, hfield=field))
    surfaces.append((x0, x1, y0, y1, normalized.astype(float) * span + zmin))

  # Compute support heights from the same triangles used for collision/raycast.
  for points in (origin[None], goals):
    for point in points:
      heights = [
        _surface_height(point[0], point[1], x0, x1, y0, y1, elevation)
        for x0, x1, y0, y1, elevation in surfaces
        if x0 - 1e-8 <= point[0] <= x1 + 1e-8 and y0 - 1e-8 <= point[1] <= y1 + 1e-8
      ]
      if not heights:
        raise ValueError("Route point has no supporting terrain surface.")
      point[2] = max(heights)
  return rough_geometries

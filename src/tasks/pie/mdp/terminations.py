from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg


class grounded_bad_orientation:
  """Allow airborne rotations and brief ground contact during vertical jumps."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    if not 0 < cfg.params["limit_angle"] < math.pi:
      raise ValueError("limit_angle must be between zero and pi.")
    if cfg.params["duration_s"] <= 0:
      raise ValueError("duration_s must be positive.")
    self._steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._last_step = torch.full_like(self._steps, -1)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self._steps[env_ids] = 0
    self._last_step[env_ids] = -1

  def __call__(
    self, env: ManagerBasedRlEnv, limit_angle: float, duration_s: float,
    sensor_names: tuple[str, ...],
  ) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    tilted = -gravity[:, 2] < math.cos(limit_angle)
    grounded = torch.zeros_like(tilted)
    for name in sensor_names:
      # Use current contacts, not history: takeoff must clear the timer.
      found = env.scene[name].data.found
      assert found is not None
      grounded |= found.reshape(env.num_envs, -1).any(dim=-1)
    fresh = self._last_step != env.common_step_counter
    updated = torch.where(tilted & grounded, self._steps + 1, 0)
    self._steps.copy_(torch.where(fresh, updated, self._steps))
    self._last_step.fill_(env.common_step_counter)
    return self._steps >= math.ceil(duration_s / env.step_dt)


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate when a non-foot terrain contact exceeds the force threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_magnitude = torch.norm(data.force_history, dim=-1)
    return (force_magnitude > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)

"""PIE environment with a single noisy proprioceptive sample per step."""

from dataclasses import dataclass

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@dataclass(kw_only=True)
class PIEEnvCfg(ManagerBasedRlEnvCfg):
  depth_render_period_steps: int = 1
  """Render the depth camera every N control steps (50 Hz / N).

  The policy consumes depth through a 10 Hz frame history, so rendering at
  the full 50 Hz control rate wastes most renders. Period 5 matches the
  RealSense-grade 10 Hz camera clock used by the PIE paper: frames become
  up to N-1 control steps stale, exactly like a physical camera whose clock
  is not aligned with the control loop or episode resets. Keep 1 for
  playback so the depth display refreshes at the sensor rate.
  """

  @property
  def class_type(self):
    return PIEEnv


class PIEEnv(ManagerBasedRlEnv):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._install_depth_render_schedule()

  def _install_depth_render_schedule(self) -> None:
    """Skip camera renders between depth ticks; raycasts stay at full rate."""
    period = int(self.cfg.depth_render_period_steps)
    ctx = self.sim._sensor_context
    if period <= 1 or ctx is None or not ctx.has_cameras:
      return
    sim = self.sim
    full_sense = sim.sense
    wp_model, wp_data = sim._wp_model, sim._wp_data
    device = sim.wp_device
    tick = -1

    def scheduled_sense() -> None:
      nonlocal tick
      tick += 1
      if tick % period == 0:
        full_sense()
        return
      ctx.prepare()
      with wp.ScopedDevice(device):
        mjwarp.refit_bvh(wp_model, wp_data, ctx.render_context)
        for sensor in ctx.raycast_sensors:
          sensor.raycast_kernel(rc=ctx.render_context)
      ctx.finalize()

    sim.sense = scheduled_sense

  def _sync_proprioception(self, observations):
    """Use the latest samples of the term-major history as current input.

    ObservationManager owns history advancement and reset/backfill. Deriving
    the current vector here avoids independently sampled actor/history noise.
    The returned dictionary is also the manager's observation cache.
    """
    history = observations["proprio_history"]
    length = self.cfg.observations["proprio_history"].history_length
    sizes = self.observation_manager.group_obs_term_dim["proprio_history"]
    offset = 0
    latest = []
    for (size,) in sizes:
      term = history[:, offset:offset + size].reshape(self.num_envs, length, -1)
      latest.append(term[:, -1])
      offset += size
    observations["actor"] = torch.cat(latest, dim=-1)

  def reset(self, *args, **kwargs):
    observations, extras = super().reset(*args, **kwargs)
    self._sync_proprioception(observations)
    return observations, extras

  def step(self, actions):
    self.extras.pop("pie_episode", None)
    self._in_step = True
    try:
      result = super().step(actions)
    finally:
      self._in_step = False
    self._sync_proprioception(result[0])
    return result

  def _reset_idx(self, env_ids=None):
    if getattr(self, "_in_step", False):
      # Capture diagnostics before curriculum/reset moves the robot. The
      # distance criterion is the same traversal proxy used by the curriculum.
      terrain = self.scene.terrain
      robot = self.scene["robot"]
      distance = torch.linalg.vector_norm(
        robot.data.root_link_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2],
        dim=-1,
      )
      self.extras["pie_episode"] = {
        "env_ids": env_ids.clone(),
        "distance": distance,
        "terminated": self.termination_manager.terminated[env_ids].clone(),
        "steps": self.episode_length_buf[env_ids].clone(),
        "terrain_level": terrain.terrain_levels[env_ids].clone(),
        "terrain_column": terrain.terrain_types[env_ids].clone(),
      }
    super()._reset_idx(env_ids)

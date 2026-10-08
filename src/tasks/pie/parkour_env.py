"""PIE environment with ordered-route episode diagnostics."""

from dataclasses import dataclass

import torch

from .env import PIEEnv, PIEEnvCfg


@dataclass(kw_only=True)
class PIEParkourEnvCfg(PIEEnvCfg):
  @property
  def class_type(self):
    return PIEParkourEnv


class PIEParkourEnv(PIEEnv):
  def reset(self, *, seed=None, env_ids=None, options=None):
    """Compute guidance from the reset pose before backfilling observations."""
    del options
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()
    self.command_manager.compute(dt=0.0)
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self._sync_proprioception(self.obs_buf)
    return self.obs_buf, self.extras

  def _reset_idx(self, env_ids=None):
    route_info = None
    if getattr(self, "_in_step", False):
      route = self.command_manager.get_term("twist")
      route_info = {
        "route_success": (route.completed[env_ids]
                          & ~self.termination_manager.terminated[env_ids]).clone(),
        "goals_reached": route.goal_index[env_ids].clone(),
        "route_fraction": (route.goal_index[env_ids].float() / route.num_goals).clone(),
      }
    super()._reset_idx(env_ids)
    if route_info is not None:
      self.extras["pie_episode"].update(route_info)

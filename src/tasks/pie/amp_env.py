"""PIE Parkour with AMP transitions and per-leg episode diagnostics."""

from dataclasses import dataclass, field

import numpy as np
import torch

from src.tasks.amp_loco.mdp.observations import amp_state
from src.tasks.amp_loco.rl.amp_features import AMP_FEATURE_LAYOUT
from src.tasks.amp_loco.rl.motion_loader import Go2MotionLoader, JOINT_NAMES, MotionCfg

from .parkour_env import PIEParkourEnv, PIEParkourEnvCfg


@dataclass(kw_only=True)
class PIEParkourAMPEnvCfg(PIEParkourEnvCfg):
  motion: MotionCfg = field(default_factory=lambda: MotionCfg(preload_transitions=False))
  amp_load_expert_data: bool = True
  amp_terrain_weights: dict[str, float] = field(default_factory=lambda: {
    "flat": 0.04,
    "slope_up": 0.02, "slope_down": 0.02,
    "stairs_up": 0.01, "stairs_down": 0.01, "step": 0.01,
    "gap": 0.004, "hurdle": 0.004, "platform": 0.004,
  })

  @property
  def class_type(self):
    return PIEParkourAMPEnv


class PIEParkourAMPEnv(PIEParkourEnv):
  """Keep AMP state pairs within an episode, including its terminal step."""

  def load_managers(self):
    robot = self.scene["robot"]
    _, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(names) != JOINT_NAMES or tuple(robot.joint_names) != JOINT_NAMES:
      raise ValueError("PIE AMP actions and NPZ data require FL/FR/RL/RR joint order")
    self.motion_dataset = (
      Go2MotionLoader(self.cfg.motion, self.step_dt, self.device)
      if self.cfg.amp_load_expert_data else None
    )
    body_names = tuple(self.cfg.motion.body_names)
    self.amp_observation_spec = {
      "version": 1, "body_names": body_names, "anchor_name": self.cfg.motion.anchor_name,
      "layout": AMP_FEATURE_LAYOUT, "dimension": 15 * len(body_names),
    }
    self.amp_body_ids, resolved = robot.find_bodies(body_names, preserve_order=True)
    if tuple(resolved) != body_names:
      raise ValueError("PIE AMP model bodies must match the configured NPZ body order")
    anchors, resolved = robot.find_bodies(self.cfg.motion.anchor_name)
    if resolved != [self.cfg.motion.anchor_name]:
      raise ValueError("PIE AMP anchor must resolve to exactly one body")
    self.amp_anchor_body_id = anchors[0]
    self._terminal_amp_state = torch.zeros(
      self.num_envs, 15 * len(body_names), device=self.device,
    )
    self._terminal_amp_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._capture_terminal = False
    self._initialize_amp_terrain_mapping()

    # Contact sensors resolve patterns in robot geom order, not pattern order.
    sensor = self.scene["feet_ground_contact"]
    _, contact_geoms = robot.find_geoms(sensor.cfg.primary.pattern)
    self.amp_foot_names = ("FL", "FR", "RL", "RR")
    expected = tuple(f"{foot}_foot_collision" for foot in self.amp_foot_names)
    if set(contact_geoms) != set(expected) or sensor.cfg.num_slots != 1:
      raise ValueError("PIE AMP diagnostics require one contact slot per Go2 foot")
    self._foot_contact_indices = [contact_geoms.index(name) for name in expected]
    self._episode_contact_steps = torch.zeros(self.num_envs, 4, device=self.device)
    self._episode_air_time = torch.zeros_like(self._episode_contact_steps)
    self._episode_max_air_time = torch.zeros_like(self._episode_contact_steps)
    self._episode_sample_count = torch.zeros(self.num_envs, device=self.device)
    self._amp_diagnostic_keys: set[str] = set()
    super().load_managers()

  def _initialize_amp_terrain_mapping(self):
    terrain = self.scene.terrain
    generator = terrain.cfg.terrain_generator
    if generator is None or not generator.curriculum:
      raise ValueError("PIE AMP requires a terrain generator with terrain types by column")
    self.amp_terrain_names = tuple(generator.sub_terrains)
    proportions = np.asarray([cfg.proportion for cfg in generator.sub_terrains.values()])
    column_types = np.searchsorted(
      np.cumsum(proportions / proportions.sum()),
      np.arange(generator.num_cols) / generator.num_cols + 0.001,
      side="right",
    )
    self._amp_column_types = torch.as_tensor(column_types, dtype=torch.long, device=self.device)
    weights = []
    for name in self.amp_terrain_names:
      weight = self.cfg.amp_terrain_weights[name]
      if not np.isfinite(weight) or weight < 0:
        raise ValueError(f"AMP terrain weight for {name!r} must be finite and nonnegative")
      weights.append(weight)
    self._amp_type_weights = torch.tensor(weights, device=self.device)

  def step(self, actions):
    # Store the terrain before curriculum/reset changes the episode assignment.
    terrain_types = self._amp_column_types[self.scene.terrain.terrain_types].clone()
    terrain_weights = self._amp_type_weights[terrain_types]
    self._terminal_amp_mask.zero_()
    for key in self._amp_diagnostic_keys:
      self.extras.get("log", {}).pop(key, None)
    self._amp_diagnostic_keys.clear()
    self._capture_terminal = True
    try:
      result = super().step(actions)
    finally:
      self._capture_terminal = False
    # Terminal environments were counted before reset; do not count their new pose.
    alive = (~self._terminal_amp_mask).nonzero(as_tuple=False).flatten()
    self._update_foot_diagnostics(alive)
    self.extras.update({
      "terminal_amp_state": self._terminal_amp_state,
      "terminal_amp_mask": self._terminal_amp_mask,
      "amp_terrain_type": terrain_types,
      "amp_terrain_weight": terrain_weights,
      "amp_terrain_names": self.amp_terrain_names,
    })
    return result

  def _update_foot_diagnostics(self, env_ids):
    found = self.scene["feet_ground_contact"].data.found
    if found is None:
      raise RuntimeError("PIE AMP diagnostics require feet contact observations")
    contact = found[env_ids][:, self._foot_contact_indices] > 0
    self._episode_sample_count[env_ids] += 1
    self._episode_contact_steps[env_ids] += contact.float()
    air_time = torch.where(contact, 0.0, self._episode_air_time[env_ids] + self.step_dt)
    self._episode_air_time[env_ids] = air_time
    self._episode_max_air_time[env_ids] = torch.maximum(
      self._episode_max_air_time[env_ids], air_time,
    )

  def _reset_idx(self, env_ids=None):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    diagnostics = None
    if self._capture_terminal:
      self.sim.forward()
      self._terminal_amp_state[env_ids] = amp_state(self)[env_ids]
      self._terminal_amp_mask[env_ids] = True
      self._update_foot_diagnostics(env_ids)
      counts = self._episode_sample_count[env_ids].clamp_min(1).unsqueeze(-1)
      contact_rates = self._episode_contact_steps[env_ids] / counts
      max_air_times = self._episode_max_air_time[env_ids].clone()
      terrain_types = self._amp_column_types[self.scene.terrain.terrain_types[env_ids]]
      route = self.command_manager.get_term("twist")
      success = route.completed[env_ids] & ~self.termination_manager.terminated[env_ids]
      diagnostics = {}
      for index, foot in enumerate(self.amp_foot_names):
        diagnostics[f"AMP/episode_contact_rate/{foot}"] = contact_rates[:, index]
        diagnostics[f"AMP/episode_max_air_time_s/{foot}"] = max_air_times[:, index]
      for index, name in enumerate(self.amp_terrain_names):
        mask = terrain_types == index
        if mask.any():
          diagnostics[f"Metrics/route_success/{name}"] = success[mask].float()
    super()._reset_idx(env_ids)
    if diagnostics is not None:
      self.extras["log"].update(diagnostics)
      self._amp_diagnostic_keys.update(diagnostics)
      self.extras["pie_episode"].update({
        "foot_contact_rate": contact_rates,
        "foot_max_air_time_s": max_air_times,
        "amp_terrain_type": terrain_types,
      })
    self._episode_contact_steps[env_ids] = 0
    self._episode_sample_count[env_ids] = 0
    self._episode_air_time[env_ids] = 0
    self._episode_max_air_time[env_ids] = 0

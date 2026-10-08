"""External route guidance producing ordinary body-frame velocity commands."""

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


class RouteVelocityCommand(UniformVelocityCommand):
  cfg: "RouteVelocityCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    terrain = env.scene.terrain
    if terrain is None or "route_goals" not in terrain.flat_patches:
      raise ValueError("RouteVelocityCommand requires terrain-provided route_goals.")
    self.goal_table = terrain.flat_patches["route_goals"]
    self.num_goals = self.goal_table.shape[2]
    self.goals = self.goal_table[terrain.terrain_levels, terrain.terrain_types].clone()
    self.goal_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.new_goal = torch.zeros(self.num_envs, device=self.device)
    self.reference_speed = torch.zeros(self.num_envs, device=self.device)
    self.speed_override: float | None = None
    self.velocity_override: torch.Tensor | None = None
    self._last_route_step = torch.full_like(self.goal_index, -1)
    self._rows = torch.arange(self.num_envs, device=self.device)
    self.is_flat = torch.zeros_like(self.completed)
    self.metrics["route_completion"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def current_goal(self):
    return self.goals[self._rows, self.goal_index.clamp(max=self.num_goals - 1)]

  def reset(self, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = self._rows[env_ids if env_ids is not None else slice(None)]
    terrain = self._env.scene.terrain
    self.goals[env_ids] = self.goal_table[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    self.goal_index[env_ids] = 0
    self.completed[env_ids] = False
    self.new_goal[env_ids] = 0
    self._last_route_step[env_ids] = self._env.common_step_counter
    # Terrain allocation is column-based, using mjlab's cumulative weights.
    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    weights = torch.tensor([v.proportion for v in generator.sub_terrains.values()], device=self.device)
    cutoffs = (weights / weights.sum()).cumsum(0)
    column = terrain.terrain_types[env_ids] / generator.num_cols + 0.001
    ids = (column[:, None] >= cutoffs).sum(1).clamp(max=len(names) - 1)
    self.is_flat[env_ids] = ids == names.index("flat") if "flat" in names else False
    return super().reset(env_ids)

  def refresh_route(self):
    """Advance at most one ordered support point per physics/control step."""
    step = self._env.common_step_counter
    fresh = self._last_route_step != step
    pos = self.robot.data.root_link_pos_w
    goal = self.current_goal
    previous_index = (self.goal_index - 1).clamp(min=0)
    previous = self.goals[self._rows, previous_index].clone()
    previous = torch.where(
      (self.goal_index == 0)[:, None], self._env.scene.env_origins, previous
    )
    segment = goal[:, :2] - previous[:, :2]
    unit = segment / torch.linalg.vector_norm(segment, dim=-1, keepdim=True).clamp_min(1e-6)
    offset = pos[:, :2] - goal[:, :2]
    near = torch.linalg.vector_norm(offset, dim=-1) < self.cfg.goal_radius
    # Crossing a waypoint within the route corridor also counts. Limit forward
    # overshoot so a teleport past several obstacles cannot complete the route.
    along = (offset * unit).sum(-1)
    cross = torch.abs(offset[:, 0] * unit[:, 1] - offset[:, 1] * unit[:, 0])
    crossed = (along >= 0) & (along < 0.8) & (cross < self.cfg.goal_radius)
    support = self._env.scene["feet_ground_contact"].data.found
    supported = support.reshape(self.num_envs, -1).any(-1)
    above_support = pos[:, 2] - goal[:, 2]
    landed = supported & (above_support > 0.05) & (above_support < 0.65)
    reached = fresh & ~self.completed & (near | crossed) & landed
    if self.velocity_override is not None:
      reached[:] = False
    self.new_goal[fresh] = reached[fresh].float()
    self.goal_index += reached.long()
    self.completed |= self.goal_index >= self.num_goals
    self._last_route_step[fresh] = step

  def _resample_command(self, env_ids):
    self.reference_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.speed_range
    )
    if self.speed_override is not None:
      self.reference_speed[env_ids] = self.speed_override

  def _update_command(self):
    if self.velocity_override is not None:
      self.vel_command_b.copy_(self.velocity_override)
      return
    if self.speed_override is not None:
      self.reference_speed.fill_(self.speed_override)
    direction = self.current_goal[:, :2] - self.robot.data.root_link_pos_w[:, :2]
    direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
    self.heading_target[:] = torch.atan2(direction[:, 1], direction[:, 0])
    self.heading_error[:] = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
    desired_world = torch.cat((direction * self.reference_speed[:, None],
                               torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
    body_velocity = quat_apply_inverse(self.robot.data.root_link_quat_w, desired_world)
    self.vel_command_b[:, 0] = body_velocity[:, 0].clamp(min=0, max=self.cfg.ranges.lin_vel_x[1])
    self.vel_command_b[:, 1] = body_velocity[:, 1].clamp(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[:, 2] = (self.cfg.heading_control_stiffness * self.heading_error).clamp(
      *self.cfg.ranges.ang_vel_z
    )
    self.vel_command_b[self.completed] = 0

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["route_completion"][:] = self.goal_index.float() / self.num_goals

  def set_velocity_override(self, velocity: torch.Tensor | None):
    """Use direct user velocities without injecting route labels into the actor."""
    self.velocity_override = None if velocity is None else velocity.detach().clone().expand(self.num_envs, 3)
    self._update_command()

  def create_gui(self, name, server, get_env_idx):
    # Route targets are visualized below. Direct velocity input is owned by
    # play_pie.py so a second joystick cannot silently override route guidance.
    pass

  def _debug_vis_impl(self, visualizer):
    super()._debug_vis_impl(visualizer)
    for env_id in visualizer.get_env_indices(self.num_envs):
      points = self.goals[env_id].detach().cpu().numpy()
      current = int(self.goal_index[env_id])
      for index, point in enumerate(points):
        color = (0.2, 0.8, 0.2, 0.8) if index < current else (0.2, 0.5, 1.0, 0.8)
        if index == current:
          color = (1.0, 0.5, 0.0, 1.0)
        center = point.copy()
        center[2] += 0.1
        visualizer.add_sphere(center, 0.08, color)
        if index:
          start = points[index - 1].copy()
          start[2] += 0.1
          visualizer.add_arrow(start, center, color, width=0.008)


@dataclass(kw_only=True)
class RouteVelocityCommandCfg(UniformVelocityCommandCfg):
  speed_range: tuple[float, float] = (0.3, 1.2)
  goal_radius: float = 0.4

  def __post_init__(self):
    super().__post_init__()
    if not 0 < self.speed_range[0] <= self.speed_range[1]:
      raise ValueError("Automatic route speeds must be strictly positive.")
    if self.goal_radius <= 0:
      raise ValueError("goal_radius must be positive.")

  def build(self, env):
    return RouteVelocityCommand(self, env)

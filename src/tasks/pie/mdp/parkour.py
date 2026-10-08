"""Rewards, episode boundaries and curriculum for externally guided PIE."""

import torch

from .rewards import track_body_planar_velocity


def route_term(env):
  return env.command_manager.get_term("twist")


def route_complete(env):
  route = route_term(env)
  route.refresh_route()
  return route.completed


def route_failed(env):
  route = route_term(env)
  route.refresh_route()
  pos = route.robot.data.root_link_pos_w
  origin = env.scene.env_origins
  generator = env.scene.terrain.cfg.terrain_generator
  # All routes start one metre from the tile's near edge.
  local_x = pos[:, 0] - origin[:, 0] + 1.0
  local_y = pos[:, 1] - origin[:, 1]
  outside = (local_x < 0.2) | (local_x > generator.size[0] - 0.2)
  outside |= local_y.abs() > generator.size[1] / 2 - 0.15
  # Goal-relative support height handles ascending and descending routes.
  fallen = pos[:, 2] < route.current_goal[:, 2] - 0.8
  return outside | fallen


def route_velocity_reward(env):
  route = route_term(env)
  if route.velocity_override is not None:
    return track_body_planar_velocity(env, sigma=0.25, command_name="twist")
  direction = route.current_goal[:, :2] - route.robot.data.root_link_pos_w[:, :2]
  direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
  speed = (direction * route.robot.data.root_link_lin_vel_w[:, :2]).sum(-1)
  return (speed / route.reference_speed.clamp_min(0.1)).clamp(max=1.0)


def route_goal_reward(env):
  return route_term(env).new_goal


def parkour_orientation(env):
  route = route_term(env)
  gravity = route.robot.data.projected_gravity_b
  penalty = gravity[:, :2].square().sum(-1)
  return penalty * torch.where(route.is_flat, 1.0, 0.2)


def parkour_vertical_velocity(env):
  route = route_term(env)
  penalty = route.robot.data.root_link_lin_vel_b[:, 2].square()
  return penalty * torch.where(route.is_flat, 1.0, 0.25)


def route_curriculum(env, env_ids):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  route = route_term(env)
  active = env.episode_length_buf[env_ids] > 0
  success = route.completed[env_ids] & ~env.termination_manager.terminated[env_ids]
  move_up = active & success
  move_down = active & ~success & (
    env.termination_manager.terminated[env_ids]
    | (route.goal_index[env_ids] < route.num_goals // 2)
  )
  env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
  return env.scene.terrain.terrain_levels.float().mean()

"""Velocity tracking and stability terms for AMP locomotion."""

import torch


def track_linear_velocity(env, command_name: str, std: float):
  command = env.command_manager.get_command(command_name)
  velocity = env.scene["robot"].data.root_link_lin_vel_b
  return torch.exp(-(command[:, :2] - velocity[:, :2]).square().sum(-1) / std**2)


def track_angular_velocity(env, command_name: str, std: float):
  command = env.command_manager.get_command(command_name)
  velocity = env.scene["robot"].data.root_link_ang_vel_b
  return torch.exp(-(command[:, 2] - velocity[:, 2]).square() / std**2)


def linear_velocity_z(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def angular_velocity_xy(env):
  return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(-1)


def base_height(env, target_height: float):
  height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return (height - target_height).square()


def collision(env, sensor_name: str):
  return (env.scene[sensor_name].data.found > 0).float().sum(-1)


def feet_air_time(env, sensor_name: str, command_name: str):
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  reward = ((sensor.data.last_air_time - 0.5) * sensor.compute_first_contact(env.step_dt)).sum(-1)
  return reward * (command[:, :2].norm(dim=-1) > 0.1)

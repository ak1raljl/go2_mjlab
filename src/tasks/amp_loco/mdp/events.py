"""Randomized joint resets following amp_go2's ordinary reset path."""

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def reset_joints_by_scale(
  env,
  env_ids,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  """Scale default joint angles independently and reset joint velocities to zero."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot = env.scene[asset_cfg.name]
  default_pos = robot.data.default_joint_pos
  limits = robot.data.soft_joint_pos_limits
  assert default_pos is not None and limits is not None
  joint_pos = default_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_limits = limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
  robot.write_joint_state_to_sim(
    joint_pos,
    torch.zeros_like(joint_pos),
    joint_ids=asset_cfg.joint_ids,
    env_ids=env_ids,
  )

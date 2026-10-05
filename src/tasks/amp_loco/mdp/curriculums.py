from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def terrain_columns_by_name(cfg: TerrainGeneratorCfg) -> dict[str, tuple[int, ...]]:
  """Match mjlab 1.2's curriculum column allocation, including normalization."""
  proportions = np.array([sub.proportion for sub in cfg.sub_terrains.values()])
  cumulative = np.cumsum(proportions / proportions.sum())
  column_types = np.searchsorted(
    cumulative, np.arange(cfg.num_cols) / cfg.num_cols + 0.001, side="right"
  )
  return {
    name: tuple(np.flatnonzero(column_types == index).tolist())
    for index, name in enumerate(cfg.sub_terrains)
  }


class terrain_level_mean:
  """Report one terrain type's mean level without changing its curriculum."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    terrain = env.scene.terrain
    assert terrain is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None and generator.curriculum
    # Resolve after CLI overrides, using the actual terrain generator settings.
    columns = terrain_columns_by_name(generator)[cfg.params["terrain_name"]]
    self.columns = torch.tensor(
      columns, device=terrain.terrain_types.device, dtype=torch.long
    )

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, terrain_name: str
  ) -> torch.Tensor | None:
    del env_ids, terrain_name  # Include all environments, not only the reset subset.
    terrain = env.scene.terrain
    assert terrain is not None
    mask = torch.isin(terrain.terrain_types, self.columns)
    if not mask.any():
      # An unassigned type has no meaningful level; do not log zero or NaN.
      return None
    return terrain.terrain_levels[mask].float().mean()


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])

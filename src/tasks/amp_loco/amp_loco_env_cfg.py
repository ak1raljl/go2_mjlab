"""Base AMP locomotion configuration with per-robot settings filled by adapters."""

import math
from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg,
  RewardTermCfg, TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig
from src.tasks.amp_loco import mdp
from src.tasks.amp_loco.rl.motion_loader import MotionCfg


@dataclass(kw_only=True)
class AmpLocoEnvCfg(ManagerBasedRlEnvCfg):
  motion: MotionCfg = field(default_factory=MotionCfg)
  reference_init_probability: float = 1.0
  reference_height_offset: float = 0.0  # Set per-robot.

  @property
  def class_type(self):
    from .env import AmpLocoEnv
    return AmpLocoEnv


def make_amp_loco_env_cfg() -> AmpLocoEnvCfg:
  """Create a fresh base task; robot configs supply assets, sensors, and motion data."""

  # Observations.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=True, history_length=1),
    "critic": ObservationGroupCfg(
      terms={**deepcopy(actor_terms), "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel)},
      enable_corruption=False, history_length=1,
    ),
    "amp": ObservationGroupCfg(
      terms={"state": ObservationTermCfg(func=mdp.amp_state)}, enable_corruption=False,
    ),
  }

  # Actions and commands.
  actions = {"joint_pos": JointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=0.25,  # Override per-robot.
    use_default_offset=True,
  )}
  commands = {"twist": UniformVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.05,
    heading_command=False, debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(
      # Set per-robot from the first command curriculum stage.
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
    ),
  )}

  # Events.
  events = {
    "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    "reference_init": EventTermCfg(func=mdp.reset_reference_state, mode="reset"),
  }

  # Rewards. Robot configs install the named contact sensors.
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity, weight=4.0, params={"command_name": "twist", "std": 0.5},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity, weight=2.0, params={"command_name": "twist", "std": 0.5},
    ),
    "linear_velocity_z": RewardTermCfg(func=mdp.linear_velocity_z, weight=-1.0),
    "angular_velocity_xy": RewardTermCfg(func=mdp.angular_velocity_xy, weight=-0.05),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-4),
    "base_height": RewardTermCfg(
      func=mdp.base_height, weight=-1.0, params={"target_height": 0.0},  # Set per-robot.
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "collision": RewardTermCfg(
      func=mdp.collision, weight=-1.0, params={"sensor_name": "thigh_ground_contact"},
    ),
    "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time, weight=1.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
  }

  # Terminations.
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70)}),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact, params={"sensor_name": "base_ground_contact", "force_threshold": 1.0},
    ),
  }

  # Curriculum. Robot configs set all axis ranges for each stage.
  curriculum = {"command_vel": CurriculumTermCfg(
    func=mdp.commands_vel, params={"command_name": "twist", "velocity_stages": []},
  )}

  # Scene and simulation. Robot assets and contact sensors are set per-robot.
  return AmpLocoEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"), entities={}, sensors=(),
      num_envs=1, extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    motion=MotionCfg(motion_dir="", exclude_patterns=(), body_names=(), anchor_name=""),  # Set per-robot.
    sim=SimulationCfg(
      njmax=300, nconmax=None, contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50),
    ),
    decimation=4,
    episode_length_s=20.0,
    viewer=ViewerConfig(entity_name="robot", body_name="", distance=1.5),  # Set per-robot.
  )

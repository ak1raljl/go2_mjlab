"""Flat Go2 AMP task; share robot assets, never velocity task configuration."""

from copy import deepcopy

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from src.tasks.amp_loco.amp_loco_env_cfg import AmpLocoEnvCfg, make_amp_loco_env_cfg
from src.tasks.amp_loco.rl.motion_loader import MotionCfg


def unitree_go2_amp_loco_env_cfg(play: bool = False) -> AmpLocoEnvCfg:
  """Fill the base AMP task with Go2 assets, contacts, and expert data."""
  cfg = make_amp_loco_env_cfg()
  cfg.scene.entities = {"robot": deepcopy(get_go2_robot_cfg())}

  # Go2 contact geometry.
  foot_geoms = tuple(f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR"))
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
  )
  base = ContactSensorCfg(
    name="base_ground_contact",
    primary=ContactMatch(mode="geom", pattern="base.*_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )
  thighs = ContactSensorCfg(
    name="thigh_ground_contact",
    primary=ContactMatch(mode="geom", pattern=".*_thigh_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet, base, thighs)

  # Go2 motion layout and control settings.
  cfg.motion = MotionCfg(
    preload_transitions=True,
    num_preload_transitions=1_000_000,
    preload_batch_size=16_384,
  )
  # NPZ feet centers are about 9 mm below the current 22 mm foot radius.
  cfg.reference_height_offset = 0.01
  cfg.rewards["base_height"].params["target_height"] = 0.32

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = 0.25
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.debug_vis = play

  # One PPO iteration is 24 control steps with the default runner settings.
  stages = [
    {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.5, 0.5)},
    {"step": 1000 * 24, "lin_vel_x": (-0.8, 1.5), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.75, 0.75)},
    {"step": 3000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.0, 1.0)},
    {"step": 5000 * 24, "lin_vel_x": (-1.2, 3.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-1.0, 1.0)},
  ]
  cfg.curriculum["command_vel"].params["velocity_stages"] = stages
  for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    setattr(command.ranges, axis, stages[0][axis])

  # Playback overrides.
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.reference_init_probability = 0.0
    cfg.motion.preload_transitions = False
    cfg.curriculum = {}
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      setattr(command.ranges, axis, stages[-1][axis])
  return cfg

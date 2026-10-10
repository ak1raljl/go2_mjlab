"""Go2 PIE on externally guided Extreme Parkour / PIE route terrains."""

import copy
import math
from dataclasses import fields, replace

from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.pie import mdp
from src.tasks.pie.mdp import parkour
from src.tasks.pie.mdp.route_command import RouteVelocityCommandCfg
from src.tasks.pie.parkour_env import PIEParkourEnvCfg
from src.tasks.pie.parkour_terrains import PIE_PARKOUR_TERRAINS_CFG
from .env_cfgs import unitree_go2_pie_env_cfg


_PARKOUR_CAMERA_POSITION = (0.34, 0.0, 0.07)
_PARKOUR_CAMERA_PITCH_DEG = 30.0
_PARKOUR_CAMERA_FOVY_DEG = 67.0
_PARKOUR_CAMERA_RESOLUTION = 120


def unitree_go2_pie_parkour_env_cfg(play: bool = False) -> PIEParkourEnvCfg:
  base = unitree_go2_pie_env_cfg(play=play)
  cfg = PIEParkourEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
  # Match MGDP's runtime Go2 camera, not its URDF visual camera joint.
  # Square pixels/images make its 67-degree horizontal FOV also the vertical
  # FOV. MuJoCo camera -Z looks forward/down; +Y stays image-up.
  half_angle = math.radians(90.0 - _PARKOUR_CAMERA_PITCH_DEG) / 2.0
  c, s = math.cos(half_angle) / math.sqrt(2.0), math.sin(half_angle) / math.sqrt(2.0)
  cfg.scene.sensors = tuple(
    replace(
      sensor, pos=_PARKOUR_CAMERA_POSITION, quat=(c, s, -s, -c),
      fovy=_PARKOUR_CAMERA_FOVY_DEG,
      width=_PARKOUR_CAMERA_RESOLUTION, height=_PARKOUR_CAMERA_RESOLUTION,
    )
    if sensor.name == "front_depth" else sensor
    for sensor in cfg.scene.sensors
  )
  # Keep the complete MGDP frustum and resize to PIE's existing encoder input.
  cfg.observations["camera"].terms["front_depth"].params.update(
    crop_left=0, crop_right=0, resize=(60, 86),
  )
  if "camera_position" in cfg.events:
    cfg.events["camera_position"].params["ranges"] = {
      axis: (-0.02, 0.02) for axis in range(3)
    }
  if "camera_pitch" in cfg.events:
    cfg.events["camera_pitch"].params["pitch_range"] = (
      -math.radians(2.0), math.radians(2.0),
    )
  cfg.events.pop("camera_fovy", None)
  cfg.scene.terrain.terrain_generator = copy.deepcopy(PIE_PARKOUR_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 1
  cfg.episode_length_s = 40.0
  cfg.events.pop("randomize_terrain", None)
  cfg.commands["twist"] = RouteVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(6.0, 6.0),
    resample_on_reset_only=play,
    heading_control_stiffness=1.5, debug_vis=True,
    speed_range=(0.3, 1.2),
    ranges=RouteVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 1.5), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-1.2, 1.2),
    ),
  )
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=parkour.route_velocity_reward, weight=1.5,
  )
  cfg.rewards["route_goal"] = RewardTermCfg(func=parkour.route_goal_reward, weight=2.0)
  cfg.rewards["orientation"] = RewardTermCfg(func=parkour.parkour_orientation, weight=-1.0)
  cfg.rewards["lin_vel_z"] = RewardTermCfg(func=parkour.parkour_vertical_velocity, weight=-1.0)
  # Do not impose a fixed trot phase or standing height on jumping routes.
  cfg.rewards.pop("foot_gait", None)
  cfg.rewards.pop("base_height", None)
  torso_contact = ContactSensorCfg(
    name="torso_ground_contact",
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, torso_contact)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.grounded_bad_orientation,
    params={
      "limit_angle": math.radians(85),
      "duration_s": 0.5,
      "sensor_names": ("feet_ground_contact", "nonfoot_ground_touch"),
    },
  )
  # The inherited collision reward already penalizes base collision geoms.
  # Allow recovery after torso contact without a separate contact termination.
  cfg.terminations["route_complete"] = TerminationTermCfg(
    func=parkour.route_complete, time_out=True,
  )
  cfg.terminations["route_failed"] = TerminationTermCfg(func=parkour.route_failed)
  cfg.terminations["gap_fall_height"] = TerminationTermCfg(
    func=parkour.gap_fall_height, params={"max_drop": 0.20},
  )
  cfg.curriculum = {} if play else {
    "terrain_levels": CurriculumTermCfg(func=parkour.route_curriculum),
    # Log after the single curriculum update; these terms only read levels.
    **{
      f"terrain_levels/{name}": CurriculumTermCfg(
        func=mdp.terrain_level_mean, params={"terrain_name": name},
      )
      for name in cfg.scene.terrain.terrain_generator.sub_terrains
    },
  }
  if play:
    # Keep one column per kind so random playback never omits a terrain family.
    cfg.scene.terrain.terrain_generator.num_cols = len(
      cfg.scene.terrain.terrain_generator.sub_terrains
    )
    for terrain_cfg in cfg.scene.terrain.terrain_generator.sub_terrains.values():
      terrain_cfg.proportion = 1.0
  return cfg

"""Independent Go2 AMP configurations for flat and rough terrain."""

from copy import deepcopy

from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg

from src.tasks.amp_loco import mdp
from src.tasks.amp_loco.amp_loco_env_cfg import AmpLocoEnvCfg, make_amp_loco_env_cfg
from src.tasks.amp_loco.rl.motion_loader import MotionCfg
from .terrains import make_amp_rough_terrains_cfg


def _unitree_go2_amp_loco_base_env_cfg(
  play: bool = False,
) -> AmpLocoEnvCfg:
  """Fill independent Go2 assets, sensors, rewards, and play overrides."""
  cfg = make_amp_loco_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": deepcopy(get_go2_robot_cfg())}
  # amp_go2's ordinary reset starts above the ground, with randomized joints.
  cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.42)
  cfg.motion = MotionCfg(
    preload_transitions=not play,
    num_preload_transitions=2_000_000,
    preload_batch_size=16_384,
  )

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  # cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  # cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_amp_loco_env_cfg(play: bool = False) -> AmpLocoEnvCfg:
  """Create a flat velocity-aligned task and set AMP expert data."""
  cfg = _unitree_go2_amp_loco_base_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  # del cfg.observations["actor"].terms["height_scan"]
  # del cfg.observations["critic"].terms["height_scan"]

  # Flat terrain has no terrain curriculum; play mode clears all curricula.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg


def unitree_go2_amp_rough_env_cfg(play: bool = False) -> AmpLocoEnvCfg:
  """Create an AMP rough task with privileged terrain scan and ordinary resets."""
  cfg = _unitree_go2_amp_loco_base_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=make_amp_rough_terrains_cfg(play=play),
    max_init_terrain_level=5,
  )
  if not play:
    # The existing terrain_levels term updates each environment once. These
    # scalar terms only report levels, preserving the aggregate log and the
    # curriculum manager's viewer support in mjlab 1.2.
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    for name in generator.sub_terrains:
      cfg.curriculum[f"terrain_levels/{name}"] = CurriculumTermCfg(
        func=mdp.terrain_level_mean,
        params={"terrain_name": name},
      )
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-1.0, 1.0), "y": (-1.0, 1.0),
  }
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    RayCastSensorCfg(
      name="terrain_scan",
      frame=ObjRef(type="body", name="base_link", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
      max_distance=5.0,
      include_geom_groups=(0,),
      exclude_parent_body=True,
      debug_vis=True,
    ),
  )
  # Match amp_go2's privileged scan: clip(base_z - ground_z - 0.5, -1, 1) * 2.5.
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan", "offset": 0.5},
    clip=(-1.0, 1.0),
    scale=2.5,
  )

  # Feet need clearance above local terrain rather than absolute world Z.
  foot_sensor_names = tuple(f"foot_terrain_{foot}" for foot in ("FR", "FL", "RR", "RL"))
  cfg.scene.sensors += tuple(
    RayCastSensorCfg(
      name=name,
      frame=ObjRef(type="site", name=foot, entity="robot"),
      ray_alignment="world",
      pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
      max_distance=1.0,
      include_geom_groups=(0,),
    )
    for name, foot in zip(foot_sensor_names, ("FR", "FL", "RR", "RL"))
  )
  cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
    func=mdp.foot_height_above_terrain,
    params={"sensor_names": foot_sensor_names},
  )

  if play:
    # Select the new patch before resetting the robot onto its spawn origin.
    randomize_terrain = cfg.events.pop("randomize_terrain")
    cfg.events = {"randomize_terrain": randomize_terrain, **cfg.events}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

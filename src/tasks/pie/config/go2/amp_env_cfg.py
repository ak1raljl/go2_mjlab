"""PIE Parkour settings with terrain-dependent AMP style rewards."""

from dataclasses import fields

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg

from src.tasks.amp_loco.mdp.observations import amp_state
from src.tasks.amp_loco.rl.motion_loader import MotionCfg
from src.tasks.pie.amp_env import PIEParkourAMPEnvCfg

from .parkour_env_cfg import unitree_go2_pie_parkour_env_cfg


def unitree_go2_pie_parkour_amp_env_cfg(play: bool = False) -> PIEParkourAMPEnvCfg:
  base = unitree_go2_pie_parkour_env_cfg(play=play)
  cfg = PIEParkourAMPEnvCfg(**{
    item.name: getattr(base, item.name) for item in fields(base) if item.init
  })
  cfg.scene.num_envs = 1 if play else 2048
  # Continuous-time sampling avoids a 2.9 GiB expert transition cache on GPU.
  cfg.motion = MotionCfg(preload_transitions=False)
  cfg.amp_load_expert_data = not play
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={"state": ObservationTermCfg(func=amp_state)},
    enable_corruption=False,
  )
  return cfg

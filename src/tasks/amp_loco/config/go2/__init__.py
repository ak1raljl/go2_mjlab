from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl.runner import AmpOnPolicyRunner
from .env_cfgs import unitree_go2_amp_loco_env_cfg, unitree_go2_amp_rough_env_cfg
from .rl_cfg import unitree_go2_amp_loco_runner_cfg, unitree_go2_amp_rough_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-AMP-Loco",
  env_cfg=unitree_go2_amp_loco_env_cfg(),
  play_env_cfg=unitree_go2_amp_loco_env_cfg(play=True),
  rl_cfg=unitree_go2_amp_loco_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-AMP-Rough",
  env_cfg=unitree_go2_amp_rough_env_cfg(),
  play_env_cfg=unitree_go2_amp_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_amp_rough_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

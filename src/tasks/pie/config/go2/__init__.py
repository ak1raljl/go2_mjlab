"""Register the Unitree Go2 PIE task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.pie.rl import PIEOnPolicyRunner

from .env_cfgs import unitree_go2_pie_env_cfg
from .rl_cfg import unitree_go2_pie_ppo_runner_cfg
from .parkour_env_cfg import unitree_go2_pie_parkour_env_cfg


register_mjlab_task(
  task_id="Unitree-Go2-PIE",
  env_cfg=unitree_go2_pie_env_cfg(),
  play_env_cfg=unitree_go2_pie_env_cfg(play=True),
  rl_cfg=unitree_go2_pie_ppo_runner_cfg(),
  runner_cls=PIEOnPolicyRunner,
)


def _parkour_runner_cfg():
  cfg = unitree_go2_pie_ppo_runner_cfg()
  cfg.experiment_name = "go2_pie_parkour"
  return cfg


register_mjlab_task(
  task_id="Unitree-Go2-PIE-Parkour",
  env_cfg=unitree_go2_pie_parkour_env_cfg(),
  play_env_cfg=unitree_go2_pie_parkour_env_cfg(play=True),
  rl_cfg=_parkour_runner_cfg(),
  runner_cls=PIEOnPolicyRunner,
)

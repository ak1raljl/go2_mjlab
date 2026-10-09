"""PIE Parkour training with the AMP Rough discriminator architecture."""

from dataclasses import asdict, dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.amp_loco.config.go2.rl_cfg import AmpCfg
from src.tasks.pie.rl.config import PIEPpoAlgorithmCfg

from .rl_cfg import unitree_go2_pie_ppo_runner_cfg


@dataclass
class PIEAmpCfg(AmpCfg):
  # Expert transitions are sampled on demand by the environment's motion
  # loader. Keep the policy replay bounded alongside the visual rollout.
  amp_replay_buffer_size: int = 100_000
  amp_reward_coef: float = 1.0
  # Request the raw style score; PIEAMPPPO applies the terrain-specific
  # additive weight while retaining the complete task reward.
  amp_task_reward_lerp: float = 0.0


@dataclass
class PIEAmpPpoAlgorithmCfg(PIEPpoAlgorithmCfg):
  class_name: str = "src.tasks.pie.rl.amp_ppo:PIEAMPPPO"
  amp_cfg: PIEAmpCfg = field(default_factory=PIEAmpCfg)


def unitree_go2_pie_parkour_amp_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Add adversarial style learning without changing PIE policy inputs."""
  cfg = unitree_go2_pie_ppo_runner_cfg()
  algorithm = asdict(cfg.algorithm)
  algorithm["class_name"] = "src.tasks.pie.rl.amp_ppo:PIEAMPPPO"
  cfg.algorithm = PIEAmpPpoAlgorithmCfg(**algorithm)
  cfg.obs_groups["amp"] = ("amp",)
  cfg.experiment_name = "go2_pie_parkour_amp"
  return cfg

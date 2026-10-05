"""AMP-specific configuration using the rsl_rl_amp amp_cfg contract."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpCfg:
  expert_data: dict[str, str] = field(default_factory=lambda: {
    "class_name": "src.tasks.amp_loco.rl.motion_loader:expert_data_from_env",
  })
  amp_discr_hidden_dims: tuple[int, ...] = (1024, 512)
  amp_replay_buffer_size: int = 500_000
  amp_reward_coef: float = 0.2
  amp_task_reward_lerp: float = 0.8
  amp_loss_coef: float = 1.0
  amp_grad_pen_coef: float = 10.0
  amp_trunk_weight_decay: float = 1e-3
  amp_head_weight_decay: float = 1e-1
  min_normalized_std: float | None = 0.01
  normalizer_epsilon: float = 1e-4
  normalizer_clip_obs: float = 10.0


@dataclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "src.tasks.amp_loco.rl.amp_ppo:AMPPPO"
  amp_cfg: AmpCfg = field(default_factory=AmpCfg)


def unitree_go2_amp_loco_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Configure independent AMP models with velocity-aligned PPO hyperparameters."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=AmpPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",), "amp": ("amp",)},
    experiment_name="go2_amp_loco",
    upload_model=False,
    logger="tensorboard",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=10001,
  )

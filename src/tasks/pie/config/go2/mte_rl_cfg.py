"""A fusion-only ablation of the PIE Parkour AMP policy."""

from copy import deepcopy

from mjlab.rl import RslRlOnPolicyRunnerCfg

from .amp_rl_cfg import unitree_go2_pie_parkour_amp_runner_cfg


def unitree_go2_mte_parkour_amp_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = deepcopy(unitree_go2_pie_parkour_amp_runner_cfg())
  cfg.actor.class_name = "src.tasks.pie.rl.mte_model:MTEActorModel"
  cfg.actor.cnn_cfg = deepcopy(cfg.actor.cnn_cfg)
  for key in ("transformer_layers", "transformer_ff_dim", "transformer_dropout"):
    cfg.actor.cnn_cfg.pop(key, None)
  cfg.actor.cnn_cfg.update(
    token_dim=64, attention_heads=8, global_dim=32,
    memory_hidden_dim=128, gradient_checkpointing=True,
  )
  cfg.experiment_name = "go2_mte_parkour_amp"
  return cfg

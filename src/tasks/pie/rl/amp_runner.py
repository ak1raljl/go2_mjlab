"""PIE exports with logging of the task and adversarial style rewards."""

import json

import torch

from .runner import PIEOnPolicyRunner


class PIEAMPOnPolicyRunner(PIEOnPolicyRunner):
  """Preserve the PIE training loop and deployment contract for AMP policies."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # RSL-RL's loop logs the environment reward, before algorithm-side AMP
    # mixing. Adapt this one logger instance instead of duplicating learn().
    self._base_process_env_step = self.logger.process_env_step
    self.logger.process_env_step = self._process_logging_env_step
    self._base_log = self.logger.log
    self.logger.log = self._log_iteration

  def _log_iteration(self, *args, **kwargs):
    # RSL-RL enumerates only the first step's metric keys. Make late episode
    # diagnostics visible without fabricating zeros for steps with no resets.
    entries = self.logger.ep_extras
    if entries:
      for entry in entries[1:]:
        for key in entry:
          if key not in entries[0]:
            entries[0][key] = torch.empty(0, device=self.device)
    return self._base_log(*args, **kwargs)

  def _process_logging_env_step(
    self, rewards, dones, extras, intrinsic_rewards=None
  ):
    components = self.alg.get_reward_components()
    metrics = {
      f"Rewards/{name}": values.detach().mean()
      for name, values in components.items()
    }
    # Terrain statistics must refer to the transition's original tile, since
    # an episode reset can already have assigned the robot a different tile.
    terrain_components = getattr(self.alg, "get_terrain_reward_components", None)
    if terrain_components is not None:
      metrics.update({
        f"Rewards/terrain/{name}": values.detach().mean()
        for name, values in terrain_components().items()
      })
    mixed = self.alg.get_logging_rewards()
    if mixed is None:
      mixed = rewards
    metrics["Rewards/total"] = mixed.detach().mean()
    # Logger retains each dictionary until the iteration ends. Copy it so a
    # subsequent environment step cannot overwrite these reward samples.
    logging_extras = dict(extras)
    group = "episode" if "episode" in extras else "log"
    logging_extras[group] = {**extras.get(group, {}), **metrics}
    self._base_process_env_step(
      mixed, dones, logging_extras, intrinsic_rewards
    )

  def get_policy_metadata(self, run_path: str = "local") -> dict:
    metadata = super().get_policy_metadata(run_path)
    metadata["training_method"] = "pie-amp"
    env = self.env.unwrapped
    spec = getattr(env, "amp_observation_spec", None)
    if spec is not None:
      metadata.update({
        "amp_observation_dimension": str(spec["dimension"]),
        "amp_observation_layout": spec["layout"],
        "amp_body_names": ",".join(spec["body_names"]),
        "amp_anchor_name": spec["anchor_name"],
        "amp_terrain_weights": json.dumps(env.cfg.amp_terrain_weights),
        "amp_reward_formula": "task_reward + terrain_weight * style_score",
      })
    return metadata

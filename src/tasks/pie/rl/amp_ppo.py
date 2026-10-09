"""PIE estimation and PPO training with a terrain-weighted AMP objective."""

import torch

from src.tasks.amp_loco.rl.amp_ppo import AMPPPO
from .ppo import PIEPPO


class PIEAMPPPO(AMPPPO, PIEPPO):
  """Reuse AMP lifecycle and PIE's recurrent, single-encoder update.

  The cooperative MRO routes AMP initialization through PIEPPO so its five
  estimator coefficients and successor targets are retained. The explicit
  update uses PIE's loss graph and AMP's auxiliary optimizer hooks.
  """

  update = PIEPPO.update

  def process_env_step(self, obs, rewards, dones, extras):
    state = self._get_amp_state(self.transition.observations)
    next_state = self._get_amp_state(obs).clone()
    done_mask = dones.reshape(-1).bool()
    if done_mask.any():
      terminal = extras.get("terminal_amp_state")
      mask = extras.get("terminal_amp_mask")
      if terminal is None or mask is None or not torch.equal(mask.to(done_mask.device), done_mask):
        raise RuntimeError("PIE AMP requires pre-reset AMP states for every done environment")
      next_state[done_mask] = terminal.to(self.device)[done_mask]
    task = rewards.reshape(-1).clone()
    self.amp_replay_buffer.insert(state, next_state)
    # Request the raw [0, 1] AMP score; task reward is never rescaled.
    with torch.no_grad():
      logits = self.discriminator(torch.cat((
        self.amp_normalizer(state), self.amp_normalizer(next_state),
      ), dim=-1)).squeeze(-1)
      style = (1.0 - 0.25 * (logits - 1.0).square()).clamp_min(0.0)
      weight = extras["amp_terrain_weight"].to(self.device).reshape(-1)
      weighted = weight * style
      mixed = task + weighted
    # Calling PIE directly bypasses AMPPPO's task/style lerp while preserving
    # successor_valid handling, PPO storage, time-limit bootstrapping and reset.
    PIEPPO.process_env_step(self, obs, mixed, dones, extras)
    self._logging_rewards = mixed.detach()
    self._reward_components = {
      "task": task.detach(), "amp": style.detach(), "weighted_amp": weighted.detach(),
    }
    types = extras["amp_terrain_type"].to(self.device)
    self._terrain_reward_components = {}
    for index, name in enumerate(extras["amp_terrain_names"]):
      selected = types == index
      if selected.any():
        for key, value in self._reward_components.items():
          self._terrain_reward_components[f"{name}/{key}"] = value[selected].mean()

  def get_terrain_reward_components(self):
    return getattr(self, "_terrain_reward_components", {})

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-discriminator AMP adapted from thirdparty/rsl_rl_amp."""

from copy import deepcopy
from itertools import chain

import torch
import torch.nn.functional as F
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from .amp import AMPDataSource, AMPNormalizer
from .auxiliary_ppo import AuxiliaryPPO
from .discriminator import Discriminator
from .replay_buffer import ReplayBuffer


class AMPPPO(AuxiliaryPPO):
  """Native RSL-RL PPO with AMP auxiliary loss and mixed reward logging."""

  def __init__(
    self, actor, critic, storage, amp_data: AMPDataSource, amp_obs_groups,
    discriminator: Discriminator, amp_normalizer: AMPNormalizer,
    amp_replay_buffer_size=500_000, amp_loss_coef=1.0, amp_grad_pen_coef=10.0,
    min_normalized_std=0.01, amp_trunk_weight_decay=1e-3,
    amp_head_weight_decay=1e-1, learning_rate=1e-3, optimizer="adam", device="cpu",
    **ppo_kwargs,
  ):
    super().__init__(actor, critic, storage, learning_rate=learning_rate,
                     optimizer=optimizer, device=device, **ppo_kwargs)
    if amp_loss_coef < 0 or amp_grad_pen_coef < 0:
      raise ValueError("AMP loss and gradient penalty coefficients must be nonnegative")
    self.amp_data = amp_data
    self.amp_obs_groups = tuple(amp_obs_groups)
    self.discriminator = discriminator.to(device)
    self.amp_normalizer = amp_normalizer.to(device)
    self.amp_replay_buffer = ReplayBuffer(amp_data.observation_dim, amp_replay_buffer_size, device)
    self.amp_loss_coef = amp_loss_coef
    self.amp_grad_pen_coef = amp_grad_pen_coef
    self.min_normalized_std = min_normalized_std
    self._pending_normalizer_samples = None
    self._logging_rewards = None
    self._reward_components = {}
    self.optimizer = resolve_optimizer(optimizer)([
      {"params": list(chain(actor.parameters(), critic.parameters())), "name": "policy"},
      {"params": discriminator.trunk.parameters(), "weight_decay": amp_trunk_weight_decay, "name": "amp_trunk"},
      {"params": discriminator.amp_linear.parameters(), "weight_decay": amp_head_weight_decay, "name": "amp_head"},
    ], lr=learning_rate)

  def _get_amp_state(self, obs):
    return torch.cat([obs[group] for group in self.amp_obs_groups], dim=-1)

  def process_env_step(self, obs, rewards, dones, extras):
    state = self._get_amp_state(self.transition.observations)
    next_state = self._get_amp_state(obs).clone()
    done_mask = dones.reshape(-1).bool()
    if done_mask.any():
      terminal = extras.get("terminal_amp_state")
      mask = extras.get("terminal_amp_mask")
      if terminal is None or mask is None or not torch.equal(mask.to(done_mask.device), done_mask):
        raise RuntimeError("AMP requires pre-reset terminal states for every done environment")
      next_state[done_mask] = terminal.to(self.device)[done_mask]
    task_rewards = rewards.reshape(-1).clone()
    self.amp_replay_buffer.insert(state, next_state)
    mixed, amp, _ = self.discriminator.predict_amp_reward(state, next_state, task_rewards, self.amp_normalizer)
    super().process_env_step(obs, mixed, dones, extras)
    self._logging_rewards = mixed.detach()
    self._reward_components = {"task": task_rewards.detach(), "amp": amp.detach()}

  def _compute_auxiliary_loss(self, batch_size, batch_index):
    policy_state, policy_next = self.amp_replay_buffer.sample(batch_size, batch_index)
    expert_state, expert_next = self.amp_data.sample(batch_size, batch_index)
    normalizer = self.amp_normalizer
    policy_logits = self.discriminator(torch.cat((normalizer(policy_state), normalizer(policy_next)), dim=-1))
    expert_norm, next_norm = normalizer(expert_state), normalizer(expert_next)
    expert_logits = self.discriminator(torch.cat((expert_norm, next_norm), dim=-1))
    amp_loss = 0.5 * (F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
                      + F.mse_loss(expert_logits, torch.ones_like(expert_logits)))
    penalty = self.discriminator.compute_grad_pen(expert_norm, next_norm, self.amp_grad_pen_coef)
    self._pending_normalizer_samples = torch.cat((policy_state.detach(), expert_state.detach()))
    return self.amp_loss_coef * (amp_loss + penalty), {
      "amp": amp_loss.detach(), "amp_grad_pen": penalty.detach(),
      "amp_policy_pred": policy_logits.mean().detach(), "amp_expert_pred": expert_logits.mean().detach(),
    }

  def _after_auxiliary_update(self):
    self.amp_normalizer.update(self._pending_normalizer_samples)
    self._pending_normalizer_samples = None
    if self.min_normalized_std is not None:
      distribution = self.actor.distribution
      with torch.no_grad():
        if hasattr(distribution, "std_param"):
          distribution.std_param.clamp_(min=self.min_normalized_std)
        else:
          distribution.log_std_param.clamp_(min=float(torch.log(torch.tensor(self.min_normalized_std))))

  def train_mode(self):
    super().train_mode()
    self.discriminator.train()
    self.amp_normalizer.train()

  def eval_mode(self):
    super().eval_mode()
    self.discriminator.eval()
    self.amp_normalizer.eval()

  def save(self):
    state = super().save()
    state.update(discriminator_state_dict=self.discriminator.state_dict(),
                 amp_normalizer_state_dict=self.amp_normalizer.state_dict())
    spec = getattr(self.amp_data, "observation_spec", None)
    if spec is not None:
      state["amp_observation_spec"] = spec
    return state

  def load(self, loaded_dict, load_cfg, strict):
    if load_cfg is None:
      load_cfg = dict(actor=True, critic=True, optimizer=True, iteration=True,
                      discriminator=True, amp_normalizer=True)
    if load_cfg.get("discriminator") or load_cfg.get("amp_normalizer"):
      expected = getattr(self.amp_data, "observation_spec", None)
      if expected is not None and loaded_dict.get("amp_observation_spec") != expected:
        raise ValueError(
          "AMP checkpoint observation layout is incompatible with this task's body selection, "
          "anchor, or feature layout. Start a new run or load only actor/critic weights."
        )
    resumed = super().load(loaded_dict, load_cfg, strict)
    if load_cfg.get("optimizer"):
      # Keep the adaptive scheduler consistent with the restored Adam groups.
      self.learning_rate = self.optimizer.param_groups[0]["lr"]
    if load_cfg.get("discriminator"):
      self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
    if load_cfg.get("amp_normalizer"):
      self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
    # Replay is intentionally rebuilt from new rollouts on resume.
    return resumed

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    # The installed runner/logger expects these resolved optional configs.
    cfg["algorithm"].setdefault("rnd_cfg", None)
    cfg["algorithm"].setdefault("symmetry_cfg", None)
    algorithm = deepcopy(cfg["algorithm"])
    actor_cfg, critic_cfg = deepcopy(cfg["actor"]), deepcopy(cfg["critic"])
    alg_cls = resolve_callable(algorithm.pop("class_name"))
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    critic_cls = resolve_callable(critic_cfg.pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic", "amp"])
    amp_cfg = algorithm.pop("amp_cfg")
    data_cfg = amp_cfg.pop("expert_data")
    data = resolve_callable(data_cfg.pop("class_name"))(env=env, device=device, **data_cfg)
    dim = sum(obs[group].shape[-1] for group in cfg["obs_groups"]["amp"])
    if dim != data.observation_dim:
      raise ValueError(f"Environment/expert AMP dimensions differ: {dim}/{data.observation_dim}")
    discriminator = Discriminator(
      2 * dim, amp_cfg.pop("amp_reward_coef"), amp_cfg.pop("amp_discr_hidden_dims"),
      amp_cfg.pop("amp_task_reward_lerp"),
    )
    normalizer = AMPNormalizer(dim, amp_cfg.pop("normalizer_epsilon", 1e-4), amp_cfg.pop("normalizer_clip_obs", 10.0))
    if algorithm.pop("share_cnn_encoders", False):
      raise ValueError("AMP locomotion uses separate MLP actor/critic models")
    actor = actor_cls(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
    critic = critic_cls(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
    storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    print(f"[AMP] Actor {actor.input_dim if hasattr(actor, 'input_dim') else obs['actor'].shape[-1]}, "
          f"critic {obs['critic'].shape[-1]}, discriminator {2 * dim}")
    return alg_cls(actor, critic, storage, data, cfg["obs_groups"]["amp"], discriminator,
                   normalizer, device=device, **amp_cfg, **algorithm, multi_gpu_cfg=cfg.get("multi_gpu"))

  def broadcast_parameters(self):
    super().broadcast_parameters()
    states = [self.discriminator.state_dict(), self.amp_normalizer.state_dict()]
    torch.distributed.broadcast_object_list(states, src=0)
    self.discriminator.load_state_dict(states[0])
    self.amp_normalizer.load_state_dict(states[1])

  def reduce_parameters(self):
    parameters = list(chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters()))
    if self.rnd:
      parameters.extend(self.rnd.parameters())
    grads = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
    torch.distributed.all_reduce(grads)
    grads /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      if parameter.grad is not None:
        size = parameter.numel()
        parameter.grad.copy_(grads[offset:offset + size].view_as(parameter))
        offset += size

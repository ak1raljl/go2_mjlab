# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import autograd

if TYPE_CHECKING:
  from .amp import AMPNormalizer


class Discriminator(nn.Module):
  """Least-squares discriminator used by adversarial motion priors."""

  def __init__(
    self,
    input_dim: int,
    amp_reward_coef: float,
    hidden_layer_sizes: Sequence[int],
    task_reward_lerp: float = 0.0,
  ) -> None:
    """Build the discriminator trunk and scalar output head."""
    super().__init__()
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    if input_dim <= 0:
      raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if len(hidden_layer_sizes) == 0 or any(dim <= 0 for dim in hidden_layer_sizes):
      raise ValueError("hidden_layer_sizes must contain positive dimensions.")
    if amp_reward_coef < 0:
      raise ValueError(f"amp_reward_coef must be non-negative, got {amp_reward_coef}.")
    if not 0.0 <= task_reward_lerp <= 1.0:
      raise ValueError(f"task_reward_lerp must be in [0, 1], got {task_reward_lerp}.")

    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_layer_sizes:
      layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
      current_dim = hidden_dim
    self.trunk = nn.Sequential(*layers)
    self.amp_linear = nn.Linear(current_dim, 1)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Return discriminator logits for concatenated state transitions."""
    return self.amp_linear(self.trunk(inputs))

  def compute_grad_pen(
    self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
  ) -> torch.Tensor:
    """Penalize the discriminator gradient norm on expert transitions."""
    expert_data = torch.cat((expert_state, expert_next_state), dim=-1).detach().requires_grad_(True)
    logits = self(expert_data)
    gradients = autograd.grad(
      outputs=logits,
      inputs=expert_data,
      grad_outputs=torch.ones_like(logits),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * gradients.norm(2, dim=1).square().mean()

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer: AMPNormalizer | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mixed reward, raw AMP reward, and discriminator logits."""
    was_training = self.training
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        state = normalizer(state)
        next_state = normalizer(next_state)
      logits = self(torch.cat((state, next_state), dim=-1))
      amp_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
      amp_reward = amp_reward.squeeze(-1)
      mixed_reward = (1.0 - self.task_reward_lerp) * amp_reward + self.task_reward_lerp * task_reward
    self.train(was_training)
    return mixed_reward, amp_reward, logits

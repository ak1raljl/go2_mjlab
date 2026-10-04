# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class AMPDataSource(Protocol):
  """Interface for sampling expert state transitions used by AMP."""

  @property
  def observation_dim(self) -> int:
    """Dimension of one AMP state."""
    ...

  def sample(self, batch_size: int, batch_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of current and next expert states."""
    ...


class AMPNormalizer(nn.Module):
  """Running mean and variance normalization for AMP observations."""

  def __init__(self, observation_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0) -> None:
    """Initialize running statistics for flat AMP observations."""
    super().__init__()
    if observation_dim <= 0:
      raise ValueError(f"observation_dim must be positive, got {observation_dim}.")
    if epsilon <= 0:
      raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if clip_obs <= 0:
      raise ValueError(f"clip_obs must be positive, got {clip_obs}.")

    self.observation_dim = observation_dim
    self.epsilon = epsilon
    self.clip_obs = clip_obs
    self.register_buffer("mean", torch.zeros(observation_dim))
    self.register_buffer("var", torch.ones(observation_dim))
    self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    """Normalize and clip observations using the current statistics."""
    self._validate_observations(observations)
    std = torch.sqrt(self.var + self.epsilon)
    return torch.clamp((observations - self.mean) / std, -self.clip_obs, self.clip_obs)

  @torch.no_grad()
  def update(self, observations: torch.Tensor) -> None:
    """Update running statistics from raw, unnormalized observations."""
    if not self.training or observations.numel() == 0:
      return
    self._validate_observations(observations)

    observations = observations.detach().to(device=self.mean.device, dtype=self.mean.dtype)
    moments = observations.double()
    batch_count = torch.tensor(float(observations.shape[0]), device=self.count.device, dtype=self.count.dtype)
    total = moments.sum(dim=0)
    squares = moments.square().sum(dim=0)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      torch.distributed.all_reduce(batch_count)
      torch.distributed.all_reduce(total)
      torch.distributed.all_reduce(squares)
    batch_mean = (total / batch_count).to(self.mean.dtype)
    batch_var = (squares / batch_count - (total / batch_count).square()).clamp_min(0).to(self.var.dtype)

    count = self.count.to(dtype=self.mean.dtype)
    total_count = count + batch_count.to(dtype=self.mean.dtype)
    delta = batch_mean - self.mean
    new_mean = self.mean + delta * batch_count.to(dtype=self.mean.dtype) / total_count

    mean_accumulator = self.var * count
    batch_accumulator = batch_var * batch_count.to(dtype=self.mean.dtype)
    correction = delta.square() * count * batch_count.to(dtype=self.mean.dtype) / total_count
    new_var = (mean_accumulator + batch_accumulator + correction) / total_count

    self.mean.copy_(new_mean)
    self.var.copy_(torch.clamp_min(new_var, 0.0))
    self.count.add_(batch_count)

  def _validate_observations(self, observations: torch.Tensor) -> None:
    if observations.ndim != 2:
      raise ValueError(f"AMP observations must be 2D, got shape {tuple(observations.shape)}.")
    if observations.shape[-1] != self.observation_dim:
      raise ValueError(
        f"Expected AMP observation dimension {self.observation_dim}, got {observations.shape[-1]}."
      )

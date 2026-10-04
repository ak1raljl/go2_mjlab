# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch


class ReplayBuffer:
  """Fixed-size circular buffer for flat state transitions."""

  def __init__(self, observation_dim: int, buffer_size: int, device: str = "cpu") -> None:
    """Allocate transition storage on the requested device."""
    if observation_dim <= 0:
      raise ValueError(f"observation_dim must be positive, got {observation_dim}.")
    if buffer_size <= 0:
      raise ValueError(f"buffer_size must be positive, got {buffer_size}.")

    self.observation_dim = observation_dim
    self.buffer_size = buffer_size
    self.device = device
    self.states = torch.zeros(buffer_size, observation_dim, device=device)
    self.next_states = torch.zeros(buffer_size, observation_dim, device=device)
    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    """Insert a batch of current and next states."""
    self._validate_batch(states, next_states)
    states = states.detach().to(self.device)
    next_states = next_states.detach().to(self.device)

    if states.shape[0] >= self.buffer_size:
      self.states.copy_(states[-self.buffer_size :])
      self.next_states.copy_(next_states[-self.buffer_size :])
      self.step = 0
      self.num_samples = self.buffer_size
      return

    first_count = min(states.shape[0], self.buffer_size - self.step)
    second_count = states.shape[0] - first_count
    self.states[self.step : self.step + first_count].copy_(states[:first_count])
    self.next_states[self.step : self.step + first_count].copy_(next_states[:first_count])
    if second_count > 0:
      self.states[:second_count].copy_(states[first_count:])
      self.next_states[:second_count].copy_(next_states[first_count:])

    self.step = (self.step + states.shape[0]) % self.buffer_size
    self.num_samples = min(self.buffer_size, self.num_samples + states.shape[0])

  def sample(self, batch_size: int, batch_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample transitions with replacement."""
    del batch_index
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if self.num_samples == 0:
      raise RuntimeError("Cannot sample from an empty replay buffer.")
    indices = torch.randint(self.num_samples, (batch_size,), device=self.states.device)
    return self.states[indices], self.next_states[indices]

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    """Yield replay samples using the legacy generator interface."""
    for batch_index in range(num_mini_batch):
      yield self.sample(mini_batch_size, batch_index)

  def _validate_batch(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    if states.ndim != 2 or next_states.ndim != 2:
      raise ValueError("Replay buffer states must be 2D tensors.")
    if states.shape != next_states.shape:
      raise ValueError(f"State shapes must match, got {states.shape} and {next_states.shape}.")
    if states.shape[-1] != self.observation_dim:
      raise ValueError(
        f"Expected state dimension {self.observation_dim}, got {states.shape[-1]}."
      )

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP runner with mixed reward logging and independent policy exports."""

import os
import time
from pathlib import Path

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from rsl_rl.utils import check_nan


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    super().save(path, infos)
    directory = str(Path(path).parent)
    self.export_policy_to_onnx(directory, "policy.onnx")
    metadata = get_base_metadata(self.env.unwrapped, "local")
    group = self.env.unwrapped.cfg.observations["actor"]
    spec = self.env.unwrapped.motion_dataset.observation_spec
    metadata.update({
      "observation_history_layout": "term-major" if group.flatten_history_dim else "frame-major",
      "observation_history_length": str(group.history_length),
      "amp_observation_dimension": str(spec["dimension"]),
      "amp_observation_layout": spec["layout"],
      "amp_body_names": ",".join(spec["body_names"]),
      "amp_anchor_name": spec["anchor_name"],
    })
    attach_metadata_to_onnx(str(Path(directory) / "policy.onnx"), metadata)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None or load_cfg.get("iteration", False):
      self.current_learning_iteration += 1
    return infos

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Run the learning loop for the specified number of iterations."""
    # Randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    # Start learning
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()  # switch to train mode (for dropout for example)

    # Ensure all parameters are in-synced
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    # Initialize the logging writer
    self.logger.init_logging_writer()

    # Start training
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      # Rollout
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          # Sample actions
          actions = self.alg.act(obs)
          # Step the environment
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          # Check for NaN values from the environment
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          # Move to device
          obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
          # Process the step
          self.alg.process_env_step(obs, rewards, dones, extras)
          # Extract intrinsic rewards if RND is used (only for logging)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
          logging_rewards_fn = getattr(self.alg, "get_logging_rewards", None)
          reward_components_fn = getattr(self.alg, "get_reward_components", None)
          components = reward_components_fn() if reward_components_fn is not None else {}
          extras.setdefault("log", {}).update({
            f"Rewards/{name}": values.mean() for name, values in components.items()
          })
          # Book keeping
          self.logger.process_env_step(
            logging_rewards_fn() if logging_rewards_fn is not None else rewards,
            dones,
            extras,
            intrinsic_rewards,
          )

        stop = time.time()
        collect_time = stop - start
        start = stop

        # Compute returns
        self.alg.compute_returns(obs)

      # Update policy
      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      # Log information
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
      )

      # Save model
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

    # Save the final model after training and stop the logging writer
    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
      self.logger.stop_logging_writer()

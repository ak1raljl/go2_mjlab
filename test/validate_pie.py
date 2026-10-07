"""Check PIE recurrent replay, masked supervision and multi-step ONNX parity.

Runs on CPU without creating a simulator. A checkpoint is optional; without
one this checks a freshly initialized policy and exports into a temp directory.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch
import tyro
from rsl_rl.utils import split_and_pad_trajectories
from tensordict import TensorDict

from src.tasks.pie.config.go2.rl_cfg import unitree_go2_pie_ppo_runner_cfg
from src.tasks.pie.rl.pie_model import PIEActorModel


@dataclass
class Config:
    checkpoint_file: str | None = None
    steps: int = 12


def observations(batch: int) -> TensorDict:
    dims = {
        "actor": 45, "proprio_history": 450, "camera": 2 * 60 * 86,
        "velocity_target": 3, "height_target": 198,
        "foot_clearance_target": 4, "successor_target": 45,
    }
    values = {name: torch.randn(batch, dim) for name, dim in dims.items()}
    values["camera"] = torch.rand(batch, dims["camera"])
    values["successor_valid"] = torch.ones(batch, 1)
    return TensorDict(values, batch_size=[batch])


def main(cfg: Config) -> None:
    if cfg.steps < 4:
        raise ValueError("Use at least four steps to cover recurrent resets.")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    runner_cfg = unitree_go2_pie_ppo_runner_cfg()
    actor_cfg = asdict(runner_cfg.actor)
    actor_cfg.pop("class_name")
    actor = PIEActorModel(observations(2), runner_cfg.obs_groups, "actor", 12, **actor_cfg)
    if cfg.checkpoint_file:
        checkpoint = torch.load(cfg.checkpoint_file, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])

    # Replaying padded sequences must reproduce the action distributions that
    # generated a rollout, including hidden-state resets in different envs.
    actor.train()
    sequence = torch.stack([observations(2) for _ in range(cfg.steps)])
    dones = torch.zeros(cfg.steps, 2, 1, dtype=torch.bool)
    dones[2, 0] = True
    dones[3, 1] = True
    expected, actions, log_probs = [], [], []
    with torch.no_grad():
        actor.reset()
        for step in range(cfg.steps):
            action = actor(sequence[step], stochastic_output=True)
            expected.append(actor.output_mean.clone())
            actions.append(action)
            log_probs.append(actor.get_output_log_prob(action))
            actor.reset(dones[step, :, 0])
        padded, masks = split_and_pad_trajectories(sequence, dones)
        hidden = torch.zeros(actor.memory_num_layers, masks.shape[1], actor.memory_hidden_dim)
        actor(padded, masks=masks, hidden_state=hidden, stochastic_output=True)
        torch.testing.assert_close(actor.output_mean, torch.stack(expected), atol=2e-5, rtol=2e-5)
        replay_log_probs = actor.get_output_log_prob(torch.stack(actions))
        torch.testing.assert_close(replay_log_probs, torch.stack(log_probs), atol=2e-5, rtol=2e-5)
    print("PASS: recurrent replay and unchanged-policy log probabilities")

    # Reset successors and padded timesteps must not supervise the decoder.
    padded["successor_valid"].zero_()
    actor.zero_grad()
    losses = actor.auxiliary_losses(padded, masks=masks, hidden_state=hidden)
    assert losses["successor"].item() == 0.0
    sum(losses.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.successor_decoder.parameters())
    print("PASS: terminal/padding mask and finite auxiliary gradients")

    actor.eval()
    export = actor.as_onnx().eval()
    with tempfile.TemporaryDirectory(prefix="pie_onnx_") as directory:
        path = Path(directory) / "policy.onnx"
        torch.onnx.export(
            export, export.get_dummy_inputs(), str(path), opset_version=18,
            input_names=export.input_names, output_names=export.output_names,
            dynamo=False,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        actor.reset()
        hidden_pt = torch.zeros(actor.memory_num_layers, 1, actor.memory_hidden_dim)
        hidden_ort = hidden_pt.numpy().copy()
        max_action_error = max_memory_error = 0.0
        with torch.no_grad():
            for step in range(cfg.steps):
                if step == cfg.steps // 2:
                    actor.reset()
                    hidden_pt.zero_()
                    hidden_ort.fill(0.0)
                obs = observations(1)
                inputs = (
                    obs["actor"], obs["proprio_history"],
                    obs["camera"].reshape(1, *actor.depth_shape), hidden_pt,
                )
                action_pt, hidden_pt = export(*inputs)
                # Actor inference must not require privileged targets.
                action_live = actor(obs.select("actor", "proprio_history", "camera"))
                torch.testing.assert_close(action_live, action_pt)
                feeds = dict(zip(export.input_names, [x.numpy() for x in inputs], strict=True))
                feeds["memory_h_in"] = hidden_ort
                action_ort, hidden_ort = session.run(None, feeds)
                np.testing.assert_allclose(action_pt.numpy(), action_ort, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(hidden_pt.numpy(), hidden_ort, atol=1e-4, rtol=1e-4)
                max_action_error = max(max_action_error, float(np.abs(action_pt.numpy() - action_ort).max()))
                max_memory_error = max(max_memory_error, float(np.abs(hidden_pt.numpy() - hidden_ort).max()))
        print(f"PASS: {cfg.steps}-step ONNX parity including reset; action max error={max_action_error:.3g}, memory={max_memory_error:.3g}")


if __name__ == "__main__":
    main(tyro.cli(Config))

"""CPU checks for MTE semantics, PIE compatibility, and deployment artifacts."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch
import tyro
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import split_and_pad_trajectories

import src.tasks  # noqa: F401
from src.tasks.pie.config.go2.amp_rl_cfg import unitree_go2_pie_parkour_amp_runner_cfg
from src.tasks.pie.config.go2.mte_rl_cfg import unitree_go2_mte_parkour_amp_runner_cfg
from src.tasks.pie.rl.mte_model import MTEActorModel
from src.tasks.pie.rl.pie_model import PIEActorModel
from src.tasks.pie.rl.ppo import PIEPPO
from validate_pie import observations

BASE_TASK = "Unitree-Go2-PIE-Parkour-AMP"
MTE_TASK = "Unitree-Go2-MTE-Parkour-AMP"


@dataclass
class Config:
    checkpoint_file: str | None = None
    legacy_checkpoint: str | None = None
    baseline_ref: str = "ff11de34e05d0669ea6009e704fbd1ee6792bd6b"
    steps: int = 12
    attention_heads: int = 8
    output: str | None = None
    export_dir: str | None = None


def make_actor(mte=True, heads=8, cls=None):
    cfg = (unitree_go2_mte_parkour_amp_runner_cfg() if mte
           else unitree_go2_pie_parkour_amp_runner_cfg())
    actor_cfg = copy.deepcopy(asdict(cfg.actor))
    actor_cfg.pop("class_name")
    if mte:
        actor_cfg["cnn_cfg"]["attention_heads"] = heads
    return (cls or (MTEActorModel if mte else PIEActorModel))(
        observations(2), cfg.obs_groups, "actor", 12, **actor_cfg
    )


def estimate(actor, obs, hidden, masks=None):
    return actor._estimate_tensors(
        obs["actor"], obs["proprio_history"], obs["camera"], masks=masks,
        hidden_state=hidden, update_internal=False, sample_latent=False,
    )


def validate(cfg: Config):
    if cfg.steps < 4:
        raise ValueError("At least four steps are required for reset validation.")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    results = {}
    base_cfg = asdict(unitree_go2_pie_parkour_amp_runner_cfg())
    new_cfg = unitree_go2_mte_parkour_amp_runner_cfg()
    new_cfg.actor.cnn_cfg["attention_heads"] = 16
    assert asdict(unitree_go2_pie_parkour_amp_runner_cfg()) == base_cfg
    assert {BASE_TASK, MTE_TASK, "Unitree-Go2-PIE", "Unitree-Go2-PIE-Parkour"} <= set(list_tasks())
    assert load_rl_cfg(MTE_TASK).actor.cnn_cfg["attention_heads"] == 8
    # Same factories and full serialized values, including sensors/terrain/AMP.
    def environment_signature(task, play=False):
        # Terrain EntityCfg creates a fresh default lambda in each factory.
        return re.sub(r" at 0x[0-9a-f]+>", ">", repr(asdict(load_env_cfg(task, play=play))))
    assert environment_signature(BASE_TASK) == environment_signature(MTE_TASK)
    assert environment_signature(BASE_TASK, True) == environment_signature(MTE_TASK, True)
    results["registry_and_config_isolation"] = "PASS"

    # Evaluate the actual pre-change source; this checks initialization order,
    # parameter names and outputs, rather than comparing two new instances.
    source = subprocess.check_output(
        ["git", "show", f"{cfg.baseline_ref}:src/tasks/pie/rl/pie_model.py"], text=True
    )
    namespace = {"__name__": "legacy_pie_reference"}
    exec(compile(source, "legacy_pie_reference", "exec"), namespace)
    torch.manual_seed(83)
    original = make_actor(False, cls=namespace["PIEActorModel"])
    torch.manual_seed(83)
    current = make_actor(False)
    assert original.state_dict().keys() == current.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    results["legacy_initialization"] = "PASS: bitwise state_dict equality to baseline source"
    if cfg.legacy_checkpoint:
        state = torch.load(cfg.legacy_checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        original.load_state_dict(state, strict=True)
        current.load_state_dict(state, strict=True)
        results["legacy_checkpoint"] = cfg.legacy_checkpoint
    else:
        results["legacy_checkpoint"] = "NOT RUN: no real checkpoint supplied"
    obs = observations(2)
    hidden = torch.randn(1, 2, 128)
    with torch.no_grad():
        torch.testing.assert_close(original(obs, hidden_state=hidden), current(obs, hidden_state=hidden), rtol=0, atol=0)
    assert not any("attention." in key or "position_proj" in key for key in current.state_dict())
    results["legacy_output_regression"] = "PASS: bitwise outputs"

    actor = make_actor(heads=cfg.attention_heads)
    if cfg.checkpoint_file:
        actor.load_state_dict(torch.load(cfg.checkpoint_file, map_location="cpu", weights_only=False)["actor_state_dict"], strict=True)
    assert not hasattr(actor, "cross_modal_transformer")
    assert not any("transformer" in key for key in actor.state_dict())
    assert actor.grid_xy.shape == (28, 2) and "grid_xy" not in actor.state_dict()
    torch.testing.assert_close(actor.grid_xy[:7, 0], torch.linspace(-1, 1, 7))
    assert torch.all(actor.grid_xy[:7, 1] == -1)
    for heads in (1, 4, 8, 16):
        alternative = make_actor(heads=heads)
        assert alternative.attention.num_heads == heads
    try:
        make_actor(heads=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid head count accepted")
    results["structure_heads_and_grid"] = "PASS"

    actor.train()
    sequence = torch.stack([observations(2) for _ in range(cfg.steps)])
    initial = torch.randn(1, 2, 128)
    with torch.no_grad():
        step_hidden = initial.clone()
        outputs = []
        for step in sequence:
            values, step_hidden = estimate(actor, step, step_hidden)
            outputs.append(values["latent"])
        whole, whole_hidden = estimate(actor, sequence, initial, torch.ones(cfg.steps, 2, dtype=torch.bool))
        torch.testing.assert_close(whole["latent"], torch.stack(outputs), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(whole_hidden, step_hidden, atol=2e-5, rtol=2e-5)

    dones = torch.zeros(cfg.steps, 2, 1, dtype=torch.bool)
    dones[1, 0] = True
    dones[2, 1] = True
    expected, actions, log_probs = [], [], []
    starts = torch.zeros(cfg.steps, 1, 2, 128)
    with torch.no_grad():
        actor.reset(hidden_state=initial.clone())
        for index, step in enumerate(sequence):
            starts[index] = actor.memory_hidden
            action = actor(step, stochastic_output=True)
            assert torch.isfinite(action).all() and action.shape == (2, 12)
            expected.append(actor.output_mean.clone())
            actions.append(action)
            log_probs.append(actor.get_output_log_prob(action))
            actor.reset(dones[index, :, 0])
        padded, masks = split_and_pad_trajectories(sequence, dones)
        start_mask = torch.ones(cfg.steps, 2, dtype=torch.bool)
        start_mask[1:] = dones[:-1, :, 0]
        rollout_hidden = starts.permute(2, 0, 1, 3)[start_mask.T].transpose(0, 1).contiguous()
        actor(padded, masks=masks, hidden_state=rollout_hidden, stochastic_output=True)
        torch.testing.assert_close(actor.output_mean, torch.stack(expected), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actor.get_output_log_prob(torch.stack(actions)), torch.stack(log_probs), atol=2e-5, rtol=2e-5)
    actor.reset()
    results["recurrent_sequence_padding_reset_and_distribution"] = "PASS"

    # Pure actor loss must reach every new fusion branch and the shared CNN.
    actor.zero_grad(set_to_none=True)
    actor(obs, hidden_state=hidden).square().mean().backward()
    for name in ("depth_encoder", "history_encoder", "position_proj", "global_mlp", "query_mlp", "attention", "fusion_proj"):
        grads = [p.grad for p in getattr(actor, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(g.abs().sum().item() for g in grads) > 0, name
    results["actor_gradient_paths"] = "PASS"

    # Deterministic actions are independent of privileged targets.
    changed = obs.clone()
    for key in ("velocity_target", "height_target", "foot_clearance_target", "successor_target"):
        changed[key] += 100
    with torch.no_grad():
        torch.testing.assert_close(actor(obs, hidden_state=hidden), actor(changed, hidden_state=hidden), atol=0, rtol=0)
        a = actor.auxiliary_losses(obs, hidden_state=hidden)["velocity"]
        b = actor.auxiliary_losses(changed, hidden_state=hidden)["velocity"]
        assert a != b
    results["privileged_supervision_isolation"] = "PASS"

    # Verify the actual process_env_step labels reset observations invalid,
    # including both a fall and a timeout, then verify the masked objective.
    fake = object.__new__(PIEPPO)
    fake.transition = SimpleNamespace(observations=observations(3))
    fake.successor_target_group = "successor_target"
    next_obs = observations(3)
    with patch.object(PPO, "process_env_step"):
        PIEPPO.process_env_step(fake, next_obs, torch.zeros(3), torch.tensor([True, True, False]), {"time_outs": torch.tensor([False, True, False])})
    torch.testing.assert_close(fake.transition.observations["successor_valid"], torch.tensor([[0.], [0.], [1.]]))
    torch.testing.assert_close(fake.transition.observations["successor_target"], next_obs["successor_target"])
    target_obs = fake.transition.observations
    changed_targets = target_obs.clone()
    changed_targets["successor_target"][:2] += 1000
    torch.manual_seed(12)
    loss = actor.auxiliary_losses(target_obs)["successor"]
    torch.manual_seed(12)
    other = actor.auxiliary_losses(changed_targets)["successor"]
    torch.testing.assert_close(loss, other, atol=0, rtol=0)
    padded["successor_valid"].zero_()
    actor.zero_grad(set_to_none=True)
    losses = actor.auxiliary_losses(padded, masks=masks, hidden_state=rollout_hidden)
    assert losses["successor"] == 0
    sum(losses.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    assert all(p.grad is None or not p.grad.count_nonzero() for p in actor.successor_decoder.parameters())
    results["successor_reset_timeout_padding_mask"] = "PASS"

    # Shared estimator runs once; checkpointing must preserve all gradients.
    reference = copy.deepcopy(actor)
    reference.gradient_checkpointing = False
    actor.gradient_checkpointing = True
    gradients = []
    deterministic = []
    for model in (reference, actor):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(91)
        with patch.object(model, "_estimate_tensors", wraps=model._estimate_tensors) as call:
            action, losses = model.forward_with_auxiliary(padded, masks=masks, hidden_state=rollout_hidden)
            assert call.call_count == 1
        assert all(torch.isfinite(value) for value in losses.values())
        (action.square().mean() + sum(losses.values())).backward()
        deterministic.append(action.detach())
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(*deterministic, atol=2e-6, rtol=2e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-5)
    results["shared_forward_and_checkpoint_gradient_parity"] = "PASS"

    actor.eval()
    with tempfile.TemporaryDirectory(prefix="mte_validate_") as temporary:
        directory = Path(cfg.export_dir or temporary)
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "actor_roundtrip.pt"
        torch.save({"actor_state_dict": actor.state_dict()}, state_path)
        restored = make_actor(heads=cfg.attention_heads).eval()
        restored.load_state_dict(torch.load(state_path, weights_only=True)["actor_state_dict"], strict=True)
        export = actor.as_onnx().eval()
        with torch.no_grad():
            inputs = (obs["actor"], obs["proprio_history"], obs["camera"].reshape(2, *actor.depth_shape), hidden)
            for lhs, rhs in zip(export(*inputs), restored.as_onnx()(*inputs)):
                torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
        results["mte_checkpoint_roundtrip"] = "PASS"
        # Invoke the existing runner exporter without constructing a simulator.
        shim = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))
        MjlabOnPolicyRunner.export_policy_to_onnx(shim, str(directory))
        with torch.no_grad():
            traced = torch.jit.trace(actor.as_jit().eval(), export.get_dummy_inputs())
        traced.save(str(directory / "policy_trace.pt"))
        traced = torch.jit.load(str(directory / "policy_trace.pt"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(directory / "policy.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        hidden_pt = torch.zeros(1, 1, 128)
        hidden_jit = hidden_pt.clone()
        hidden_ort = hidden_pt.numpy().copy()
        max_errors = {"onnx_actions": 0., "onnx_hidden": 0., "jit_actions": 0., "jit_hidden": 0.}
        actor.reset()
        with torch.no_grad():
            for index in range(cfg.steps):
                if index == cfg.steps // 2:
                    hidden_pt.zero_(); hidden_jit.zero_(); hidden_ort.fill(0.)
                    actor.reset()
                item = observations(1)
                args = (item["actor"], item["proprio_history"], item["camera"].reshape(1, *actor.depth_shape))
                action_pt, next_pt = export(*args, hidden_pt)
                torch.testing.assert_close(actor(item.select("actor", "proprio_history", "camera")), action_pt)
                action_jit, hidden_jit = traced(*args, hidden_jit)
                feeds = dict(zip(export.input_names, [x.numpy() for x in (*args, hidden_pt)], strict=True))
                feeds["memory_h_in"] = hidden_ort
                action_ort, hidden_ort = session.run(None, feeds)
                for name, expected_value, actual in (
                    ("onnx_actions", action_pt.numpy(), action_ort), ("onnx_hidden", next_pt.numpy(), hidden_ort),
                    ("jit_actions", action_pt.numpy(), action_jit.numpy()), ("jit_hidden", next_pt.numpy(), hidden_jit.numpy()),
                ):
                    np.testing.assert_allclose(expected_value, actual, atol=1e-4, rtol=1e-4)
                    max_errors[name] = max(max_errors[name], float(np.abs(expected_value - actual).max()))
                hidden_pt = next_pt
        results["export_multistep_reset_max_error"] = max_errors
        results["jit_method"] = "torch.jit.trace via inherited as_jit; upstream torch.jit.script unsupported in existing PIE distribution"
    results["checkpoint_tested"] = cfg.checkpoint_file or "fresh initialization"
    if cfg.output:
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    validate(tyro.cli(Config))

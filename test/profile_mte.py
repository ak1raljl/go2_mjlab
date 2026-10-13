"""Isolated PIE/MTE parameter, latency and real rollout/update measurements.

Default: CPU structure/latency only. --training uses the normal training entry
and records full PPO + estimator + AMP updates, never synthetic training data.
"""

# ruff: noqa: E402
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tyro

from validate_mte import BASE_TASK, MTE_TASK, make_actor, observations


@dataclass
class Config:
    task: Literal["both", "pie", "mte"] = "both"
    training: bool = False
    device: str = "cpu"
    num_envs: int = 64
    iterations: int = 20
    seed: int = 42
    warmup: int = 20
    repeats: int = 100
    batch: int = 64
    attention_heads: int = 8
    gradient_checkpointing: bool = True
    output_dir: str = "outputs/mte_profile"


def gpu_snapshot():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader", "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def parameters_and_latency(cfg):
    torch.manual_seed(cfg.seed)
    actor = make_actor(cfg.task == "mte", cfg.attention_heads).to(cfg.device).eval()
    fusion_names = ("position_proj", "global_mlp", "query_mlp", "attention", "fusion_proj") if cfg.task == "mte" else ("cross_modal_transformer",)
    count = lambda module: sum(p.numel() for p in module.parameters())
    full = count(actor)
    auxiliary_only = sum(count(getattr(actor, name)) for name in ("successor_decoder", "height_decoder", "vae_logvar_head"))
    distribution = count(actor.distribution)
    result = {
        "full_actor_parameters": full,
        "fusion_parameters": {name: count(getattr(actor, name)) for name in fusion_names},
        "fusion_total": sum(count(getattr(actor, name)) for name in fusion_names),
        "auxiliary_only_parameters": auxiliary_only,
        "distribution_parameters": distribution,
        "deterministic_output_dependency_parameters": full - auxiliary_only - distribution,
        "parameter_scope": "Full actor includes both decoders, logvar head and action std. Deterministic actions/hidden exclude these; eager estimator still computes then discards logvar. Deepcopy export wrapper retains full actor; artifact size is separate.",
        "latency_ms": {},
    }
    with torch.inference_mode():
        for batch in (1, cfg.batch):
            obs = observations(batch).to(cfg.device)
            hidden = torch.zeros(1, batch, actor.memory_hidden_dim, device=cfg.device)
            for _ in range(cfg.warmup):
                actor(obs, hidden_state=hidden)
            sync(cfg.device)
            start = time.perf_counter()
            for _ in range(cfg.repeats):
                actor(obs, hidden_state=hidden)
            sync(cfg.device)
            result["latency_ms"][str(batch)] = 1000 * (time.perf_counter() - start) / cfg.repeats
    del actor
    gc.collect()
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def train_and_measure(cfg, output_dir):
    from scripts.train import TrainConfig, run_train
    import scripts.train as train_module
    from mjlab.tasks.registry import load_runner_cls

    if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Real training memory profiling requires available CUDA; run CPU structure checks without --training.")
    if cfg.num_envs % 4:
        raise ValueError("num_envs must be divisible by the baseline's four minibatches.")
    # run_train uses the same launcher environment convention.
    index = cfg.device.split(":")[-1] if ":" in cfg.device else "0"
    if int(index) != 0:
        raise ValueError("Use CUDA_VISIBLE_DEVICES=<physical GPU> with --device cuda:0.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    task = MTE_TASK if cfg.task == "mte" else BASE_TASK
    training = TrainConfig.from_task(task)
    training.env.scene.num_envs = cfg.num_envs
    training.agent.seed = cfg.seed
    training.agent.max_iterations = cfg.iterations
    training.agent.save_interval = cfg.iterations
    training.agent.run_name = "mte_profile"
    training.agent.actor.cnn_cfg = dict(training.agent.actor.cnn_cfg)
    training.agent.actor.cnn_cfg["gradient_checkpointing"] = cfg.gradient_checkpointing
    if cfg.task == "mte":
        training.agent.actor.cnn_cfg["attention_heads"] = cfg.attention_heads
    training = replace(training, enable_nan_guard=True)
    measurements = []
    runner_cls = load_runner_cls(task)

    class MeasuredRunner(runner_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            base_act, base_update = self.alg.act, self.alg.update
            first_step = True
            rollout_start = 0.0

            def measured_act(obs):
                nonlocal first_step, rollout_start
                if first_step:
                    sync(cfg.device)
                    torch.cuda.reset_peak_memory_stats(cfg.device)
                    rollout_start = time.perf_counter()
                    first_step = False
                return base_act(obs)

            def measured_update():
                nonlocal first_step
                # Match storage's split-and-pad algorithm without allocating
                # another padded visual tensor and distorting the measurement.
                dones = self.alg.storage.dones.detach().cpu().squeeze(-1).bool()
                ends = dones.clone()
                ends[-1] = True
                indices = torch.cat((torch.tensor([-1]), ends.T.reshape(-1).nonzero()[:, 0]))
                lengths = indices[1:] - indices[:-1]
                valid = dones.numel()
                padded = int(lengths.numel()) * dones.shape[0]
                sync(cfg.device)
                update_start = time.perf_counter()
                losses = base_update()
                sync(cfg.device)
                stop = time.perf_counter()
                if not all(torch.isfinite(torch.tensor(value)) for value in losses.values()):
                    raise RuntimeError(f"Nonfinite update metrics: {losses}")
                measurement = {
                    "iteration": len(measurements),
                    "update_seconds": stop - update_start,
                    "rollout_returns_update_seconds": stop - rollout_start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg.device),
                    "allocated_after_update_bytes": torch.cuda.memory_allocated(cfg.device),
                    "reserved_after_update_bytes": torch.cuda.memory_reserved(cfg.device),
                    "terminated_transitions": int(dones.sum()),
                    "valid_samples": valid, "padded_slots": padded,
                    "padding_slots": padded - valid,
                    "trajectory_count": lengths.numel(),
                    "trajectory_max_length": int(lengths.max()),
                    "trajectory_mean_length": float(lengths.float().mean()),
                    "losses": losses,
                }
                measurements.append(measurement)
                (output_dir / f"{cfg.task}_updates.json").write_text(json.dumps(measurements, indent=2) + "\n")
                first_step = True
                return losses

            self.alg.act, self.alg.update = measured_act, measured_update

        def learn(self, *args, **kwargs):
            # Model construction consumes a different amount of RNG in each
            # architecture. Re-seed so random initial episode lengths match.
            torch.manual_seed(cfg.seed)
            return super().learn(*args, **kwargs)

    train_module.load_runner_cls = lambda _: MeasuredRunner
    run_dir = output_dir / f"{cfg.task}_training"
    if run_dir.exists():
        raise FileExistsError(f"Use a fresh output directory; {run_dir} already exists.")
    run_train(task, training, run_dir)
    return {
        "iterations": measurements,
        "configuration": {
            "num_envs": cfg.num_envs, "rollout_length": training.agent.num_steps_per_env,
            "minibatches": training.agent.algorithm.num_mini_batches,
            "epochs": training.agent.algorithm.num_learning_epochs,
            "precision": "float32, no autocast; mjlab.configure_torch_backends defaults",
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "seed": cfg.seed, "episode_length_s": training.env.episode_length_s,
            "terrain_and_amp": "unchanged task factories; full config saved under training/params",
        },
        "checkpoint": str(run_dir / f"model_{cfg.iterations - 1}.pt"),
        "export_bytes": (run_dir / "policy.onnx").stat().st_size,
        "gpu_snapshot_after": gpu_snapshot(),
        "memory_scope": "PyTorch allocator peak for rollout, returns and PPO/auxiliary/AMP update; excludes export. Warp/simulator/renderer allocations outside PyTorch are excluded. nvidia-smi is a whole-device snapshot including other processes, NOT a peak nor allocator measurement. Iteration zero includes cold allocations; compare later iterations separately.",
    }


def main(cfg: Config):
    if min(cfg.iterations, cfg.num_envs, cfg.batch, cfg.repeats, cfg.warmup) <= 0:
        raise ValueError("Counts must be positive.")
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.task == "both":
        for task in ("pie", "mte"):
            command = [sys.executable, str(Path(__file__).resolve()), "--task", task]
            for key, value in asdict(cfg).items():
                if key == "task":
                    continue
                flag = key.replace("_", "-")
                if isinstance(value, bool):
                    command.append(f"--{'no-' if not value else ''}{flag}")
                else:
                    command.extend((f"--{flag}", str(value)))
            with (output_dir / f"{task}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        comparison = {task: json.loads((output_dir / f"{task}.json").read_text()) for task in ("pie", "mte")}
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"Comparison: {output_dir / 'comparison.json'}")
        return
    torch.set_num_threads(2)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process; request GPU access or use --device cpu.")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "onnxruntime")}
    tracked_diff = subprocess.check_output(["git", "diff", "HEAD"])
    sources = {}
    for pattern in ("src/tasks/pie/**/*.py", "test/*mte.py"):
        for path in Path.cwd().glob(pattern):
            sources[str(path.relative_to(Path.cwd()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": cfg.task, "settings": asdict(cfg), "versions": versions,
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(cfg.device) if cfg.device.startswith("cuda") else "CPU only",
        "gpu_snapshot_before": gpu_snapshot(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_sha256": sources,
        "structure_and_inference": parameters_and_latency(cfg),
        "training": train_and_measure(cfg, output_dir) if cfg.training else "NOT RUN: CPU/structure mode; pass --training --device cuda:0 for real training",
    }
    (output_dir / f"{cfg.task}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "training"}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))

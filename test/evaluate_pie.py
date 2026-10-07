"""Evaluate a PIE checkpoint with normal, frozen or delayed depth input.

Success is a curriculum-compatible proxy: survive an episode and finish more
than half a terrain tile from its origin. It is not a measured stair count.
"""

# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401
from scripts.play import FixedCommandController, KeyboardCommandEnv, _keyboard_command_limits


@dataclass
class Config:
    checkpoint_file: str
    output: str = "outputs/pie_evaluation.json"
    num_envs: int = 32
    steps: int = 3000
    episode_length_s: float = 20.0
    terrain_level: int = 0
    depth_mode: Literal["normal", "frozen", "delayed"] = "normal"
    depth_delay_steps: int = 5
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    seed: int = 42
    device: str = "cuda:0"


def main(cfg: Config) -> None:
    if (
        cfg.steps <= 0 or cfg.num_envs <= 0
        or cfg.episode_length_s <= 0 or cfg.depth_delay_steps < 0
    ):
        raise ValueError("Steps/environments/duration must be positive and delay nonnegative.")
    configure_torch_backends()
    env_cfg = load_env_cfg("Unitree-Go2-PIE")
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.episode_length_s
    env_cfg.observations["proprio_history"].enable_corruption = False
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)
    generator = env_cfg.scene.terrain.terrain_generator
    if not 0 <= cfg.terrain_level < generator.num_rows:
        raise ValueError(f"Terrain level must be in [0, {generator.num_rows - 1}].")
    _keyboard_command_limits(env_cfg, "twist")
    env = env_cfg.class_type(cfg=env_cfg, device=cfg.device)
    terrain = env.scene.terrain
    terrain.terrain_levels.fill_(cfg.terrain_level)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    agent_cfg = load_rl_cfg("Unitree-Go2-PIE")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = load_runner_cls("Unitree-Go2-PIE")(vec_env, asdict(agent_cfg), device=cfg.device)
        runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, map_location=cfg.device)
        policy = runner.get_inference_policy(device=cfg.device)
        controlled = KeyboardCommandEnv(vec_env, FixedCommandController(cfg.command), "twist", policy)
        controlled.initialize()
        names = list(generator.sub_terrains)
        proportions = np.array([v.proportion for v in generator.sub_terrains.values()])
        cutoffs = np.cumsum(proportions / proportions.sum())
        records = []
        episode_rewards = torch.zeros(cfg.num_envs, device=cfg.device)
        delay = deque(maxlen=cfg.depth_delay_steps + 1)
        reset_rows = torch.ones(cfg.num_envs, dtype=torch.bool, device=cfg.device)
        frozen = None
        sums = dict(
            velocity_squared_error=0.0, yaw_squared_error=0.0,
            action_delta_squared=0.0, foot_slip_squared=0.0,
        )
        previous_actions = torch.zeros(cfg.num_envs, 12, device=cfg.device)
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(cfg.steps):
                obs = controlled.get_observations()
                depth = obs["camera"].clone()
                if frozen is None:
                    frozen = depth.clone()
                frozen[reset_rows] = depth[reset_rows]
                for frame in delay:
                    frame[reset_rows] = depth[reset_rows]
                delay.append(depth)
                altered = obs.clone(recurse=False)
                if cfg.depth_mode == "frozen":
                    altered["camera"] = frozen
                elif cfg.depth_mode == "delayed":
                    altered["camera"] = delay[0]
                actions = policy(altered)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite policy action.")
                robot = env.scene["robot"]
                command = env.command_manager.get_command("twist")
                velocity_error = robot.data.root_link_lin_vel_b[:, :2] - command[:, :2]
                yaw_error = robot.data.root_link_ang_vel_b[:, 2] - command[:, 2]
                sums["velocity_squared_error"] += velocity_error.square().sum(-1).mean().item()
                sums["yaw_squared_error"] += yaw_error.square().mean().item()
                # Exclude changes across resets from action smoothness.
                previous_actions[reset_rows] = actions[reset_rows]
                sums["action_delta_squared"] += (actions - previous_actions).square().sum(-1).mean().item()
                previous_actions.copy_(actions)
                slip_cfg = env.reward_manager.get_term_cfg("foot_slip")
                sums["foot_slip_squared"] += slip_cfg.func(env, **slip_cfg.params).mean().item()
                _, reward, dones, extras = controlled.step(actions)
                episode_rewards += reward
                reset_rows = dones.bool()
                episode = extras.get("pie_episode")
                if episode is not None:
                    data = {key: value.cpu().tolist() for key, value in episode.items()}
                    for index, env_id in enumerate(data["env_ids"]):
                        column = data["terrain_column"][index]
                        terrain_id = min(
                            int(np.searchsorted(
                                cutoffs, column / generator.num_cols + 0.001, side="right"
                            )),
                            len(names) - 1,
                        )
                        failed = data["terminated"][index]
                        distance = data["distance"][index]
                        records.append({
                            "terrain": names[terrain_id], "level": data["terrain_level"][index],
                            "distance_m": distance, "failed": failed,
                            "success_proxy": not failed and distance > generator.size[0] * 0.5,
                            "duration_s": data["steps"][index] * env.step_dt,
                            "reward": episode_rewards[env_id].item(),
                        })
                    episode_rewards[reset_rows] = 0.0
        report = {
            "config": asdict(cfg),
            "success_definition": "no non-timeout termination and final planar distance > half tile length",
            "unfinished_episodes_excluded": int((env.episode_length_buf > 0).sum().item()),
            "wall_seconds": time.perf_counter() - start,
            "velocity_rmse_m_s": (sums["velocity_squared_error"] / cfg.steps) ** 0.5,
            "yaw_rmse_rad_s": (sums["yaw_squared_error"] / cfg.steps) ** 0.5,
            "action_delta_squared_mean": sums["action_delta_squared"] / cfg.steps,
            "foot_slip_squared_mean": sums["foot_slip_squared"] / cfg.steps,
            "by_terrain": {}, "episodes": records,
        }
        for name in names:
            selected = [r for r in records if r["terrain"] == name]
            report["by_terrain"][name] = {
                "completed_episodes": len(selected),
                "success_proxy_rate": (
                    np.mean([r["success_proxy"] for r in selected]).item() if selected else None
                ),
                "failure_rate": (
                    np.mean([r["failed"] for r in selected]).item() if selected else None
                ),
            }
        path = Path(cfg.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
        print(f"Saved {len(records)} completed episodes to {path}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))

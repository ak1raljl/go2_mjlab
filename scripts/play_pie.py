# ruff: noqa: E402
"""Play PIE with external route guidance, direct velocity control and depth views."""

from __future__ import annotations

import json
import copy
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# MuJoCo selects its offscreen backend at import time, before CLI parsing.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import tyro
import yaml
from mjlab.envs import mdp as env_mdp
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import src.tasks  # noqa: F401
from scripts.play import (
    FixedCommandController, KeyboardCommandEnv, KeyboardController,
    _keyboard_command_limits, _resolve_viewer,
)
from src.tasks.pie.mdp.route_command import RouteVelocityCommand
from src.tasks.pie.parkour_terrains import ParkourTerrainCfg


@dataclass(frozen=True)
class PlayPIEConfig:
    checkpoint_file: str
    task: Literal["Unitree-Go2-PIE-Parkour", "Unitree-Go2-PIE-Parkour-AMP", "Unitree-Go2-MTE-Parkour-AMP", "Unitree-Go2-PIE"] = "Unitree-Go2-PIE-Parkour"
    terrain: Literal["all", "flat", "step", "gap", "platform", "stairs_up", "stairs_down", "slope_up", "slope_down"] = "all"
    terrain_level: int = 0
    """Fixed difficulty from 0 (easy) to 9 (hard)."""
    terrain_length: float = 36.0
    """Total lane length; parkour lanes repeat roughly 18 m obstacle segments."""
    terrain_gap: float = 4.0
    """Empty lateral separation between terrain edges in metres."""
    num_envs: int = 1
    seed: int = 42
    device: str = "cuda:0"
    viewer: Literal["auto", "native", "viser"] = "auto"
    depth: bool = False
    keyboard: bool = False
    speed: float | None = None
    """Fixed route speed; omitted means sample once at each episode reset."""
    speed_range: tuple[float, float] = (0.3, 1.2)
    """Random route speed range (m/s), held until the next reset."""
    command: tuple[float, float, float] | None = None
    """Direct body vx/vy/yaw-rate override, bypassing route guidance."""
    headless_steps: int = 0
    episode_length_s: float = 120.0
    """Allow the longer route to finish even at the default minimum speed."""
    video: bool = False
    video_length: int = 200
    export: bool = False
    """Write policy.onnx with velocity/depth metadata beside the checkpoint."""
    attention_heads: int | None = None
    """MTE head count; defaults to saved params/agent.yaml, otherwise eight."""
    stats_file: str | None = None


@dataclass(kw_only=True)
class SpacedPlaybackTerrainCfg(SubTerrainCfg):
    """Repeat original obstacle segments inside a laterally separated lane."""

    terrain: SubTerrainCfg
    terrain_size: tuple[float, float]
    repetitions: int = 1

    @staticmethod
    def _edge_height(output, x: float, y: float) -> float:
        """Read centreline support at a segment seam from its actual geometry."""
        heights = []
        for item in output.geometries:
            geom = item.geom
            if geom is None:
                continue
            half = np.asarray(item.hfield.size if item.hfield is not None else geom.size)
            center = np.asarray(geom.pos)
            if not (center[0] - half[0] - 1e-6 <= x <= center[0] + half[0] + 1e-6
                    and center[1] - half[1] <= y <= center[1] + half[1]):
                continue
            if item.hfield is None:
                heights.append(center[2] + half[2])
            else:
                field = item.hfield
                samples = np.asarray(field.userdata).reshape(field.nrow, field.ncol)
                column = 0 if x <= center[0] else -1
                row = (y - center[1] + half[1]) / (2 * half[1]) * (field.nrow - 1)
                heights.append(center[2] + half[2] * np.interp(row, np.arange(field.nrow), samples[:, column]))
        if not heights:
            raise ValueError("Playback terrain segment has no support at its seam.")
        return max(heights)

    def function(self, difficulty, spec, rng):
        terrain = copy.deepcopy(self.terrain)
        length, width = self.terrain_size
        segment_length = length / self.repetitions
        terrain.size = (segment_length, width)
        geometries, patches = [], {}
        origin, previous_exit = None, None
        for index in range(self.repetitions):
            # Independent samples at the original segment scale retain obstacle
            # counts/density instead of stretching the spaces between them.
            output = terrain.function(difficulty, spec, rng)
            z_offset = 0.0
            if self.repetitions > 1:
                entry = self._edge_height(output, 0.0, width / 2)
                exit_height = self._edge_height(output, segment_length, width / 2)
                if previous_exit is not None:
                    z_offset = previous_exit - entry
                previous_exit = exit_height + z_offset
            offset = np.array([index * segment_length, (self.size[1] - width) / 2, z_offset])
            for item in output.geometries:
                if item.geom is not None:
                    item.geom.pos = np.asarray(item.geom.pos) + offset
            geometries.extend(output.geometries)
            if origin is None:
                origin = output.origin + offset
            for name, points in (output.flat_patches or {}).items():
                patches.setdefault(name, []).append(points + offset)
        return TerrainOutput(
            origin=origin, geometries=geometries,
            flat_patches={name: np.concatenate(points) for name, points in patches.items()} or None,
        )


def _configure_playback_terrain(env_cfg, cfg: PlayPIEConfig) -> None:
    generator = env_cfg.scene.terrain.terrain_generator
    if cfg.terrain != "all":
        if cfg.terrain not in generator.sub_terrains:
            raise ValueError(f"Terrain {cfg.terrain!r} is not available in {cfg.task}.")
        generator.sub_terrains = {cfg.terrain: generator.sub_terrains[cfg.terrain]}
    original_length, lane_width = generator.size
    generator.size = (cfg.terrain_length, lane_width + cfg.terrain_gap)
    terrains = {}
    for name, terrain in generator.sub_terrains.items():
        repetitions = 1
        if isinstance(terrain, ParkourTerrainCfg):
            # Leave room for six maximum-width gaps and their landing banks.
            repetitions = max(1, min(int(cfg.terrain_length / original_length + 0.5),
                                     int(cfg.terrain_length / 17.0)))
        sampling = copy.deepcopy(terrain.flat_patch_sampling)
        for patch in (sampling or {}).values():
            patch.num_patches *= repetitions
        terrains[name] = SpacedPlaybackTerrainCfg(
            terrain=terrain, terrain_size=(cfg.terrain_length, lane_width),
            repetitions=repetitions, flat_patch_sampling=sampling, proportion=1.0,
        )
    generator.sub_terrains = terrains
    generator.num_cols = len(generator.sub_terrains)
    generator.num_rows = 1
    generator.curriculum = True
    generator.difficulty_range = (cfg.terrain_level / 9, cfg.terrain_level / 9)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if "route_failed" in env_cfg.terminations:
        # Empty space separates lanes; it is not part of the drivable route.
        env_cfg.terminations["route_failed"].params["lane_width"] = lane_width


class PIEPlaybackEnv(KeyboardCommandEnv):
    """Keep direct user commands active through command-manager updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes: list[dict] = []

    def _update_command(self):
        super()._update_command()
        route = self.unwrapped.command_manager.get_term("twist")
        if self._keyboard is not None and isinstance(route, RouteVelocityCommand):
            route.set_velocity_override(self._command)

    def step(self, actions):
        route = self.unwrapped.command_manager.get_term("twist")
        speeds = route.reference_speed.clone() if isinstance(route, RouteVelocityCommand) else None
        result = super().step(actions)
        episode = result[3].get("pie_episode")
        if episode is not None:
            data = {key: value.detach().cpu().tolist() for key, value in episode.items()}
            for index in range(len(data["env_ids"])):
                entry = {key: value[index] for key, value in data.items()}
                if speeds is not None:
                    entry["reference_speed"] = float(speeds[entry["env_ids"]])
                self.episodes.append(entry)
        return result


def run_play_pie(cfg: PlayPIEConfig) -> dict:
    if cfg.num_envs <= 0 or cfg.headless_steps < 0 or cfg.episode_length_s <= 0:
        raise ValueError("Invalid environment count, step count or episode duration.")
    if not 0 <= cfg.terrain_level <= 9:
        raise ValueError("--terrain-level must be between 0 and 9.")
    if not np.isfinite(cfg.terrain_length) or cfg.terrain_length < 17.0:
        raise ValueError("--terrain-length must be finite and at least 17 metres.")
    if not np.isfinite(cfg.terrain_gap) or cfg.terrain_gap < 0:
        raise ValueError("--terrain-gap must be finite and nonnegative.")
    if not np.isfinite(cfg.speed_range).all() or not 0 < cfg.speed_range[0] <= cfg.speed_range[1] <= 1.5:
        raise ValueError("--speed-range must satisfy 0 < min <= max <= 1.5 m/s.")
    if cfg.headless_steps and (cfg.keyboard or cfg.depth):
        raise ValueError("--keyboard and --depth require interactive playback.")
    if sum((cfg.keyboard, cfg.command is not None, cfg.speed is not None)) > 1:
        raise ValueError("Choose only one of --keyboard, --command and --speed.")
    if cfg.speed is not None and not 0 < cfg.speed <= 1.5:
        raise ValueError("--speed must be in (0, 1.5] m/s.")
    if cfg.command is not None and not np.isfinite(cfg.command).all():
        raise ValueError("--command must be finite.")
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    configure_torch_backends()
    env_cfg = load_env_cfg(cfg.task, play=True)
    agent_cfg = load_rl_cfg(cfg.task)
    if cfg.task == "Unitree-Go2-MTE-Parkour-AMP":
        params_file = checkpoint.parent / "params" / "agent.yaml"
        saved_heads = None
        if params_file.is_file():
            # BaseLoader reads scalar strings only, including mjlab's tuple
            # tags; it cannot construct arbitrary Python objects from YAML.
            params = yaml.load(params_file.read_text(), Loader=yaml.BaseLoader)
            saved_heads = int(params["actor"]["cnn_cfg"]["attention_heads"])
        if cfg.attention_heads is not None and saved_heads is not None and cfg.attention_heads != saved_heads:
            raise ValueError("--attention-heads disagrees with the checkpoint's saved training config.")
        agent_cfg.actor.cnn_cfg["attention_heads"] = (
            cfg.attention_heads if cfg.attention_heads is not None else saved_heads if saved_heads is not None else 8
        )
    elif cfg.attention_heads is not None:
        raise ValueError("--attention-heads is only supported for the MTE task.")
    if cfg.task in ("Unitree-Go2-PIE-Parkour-AMP", "Unitree-Go2-MTE-Parkour-AMP"):
        from src.tasks.pie.config.go2.rl_cfg import unitree_go2_pie_ppo_runner_cfg

        # Inference needs the PIE actor only, never expert data/discriminator/replay.
        agent_cfg.algorithm = unitree_go2_pie_ppo_runner_cfg().algorithm
        agent_cfg.obs_groups.pop("amp", None)
        agent_cfg.num_steps_per_env = 1
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.episode_length_s
    _configure_playback_terrain(env_cfg, cfg)
    # Select the new tile before reset_base reads env_origins.
    env_cfg.events.pop("randomize_terrain", None)
    env_cfg.events = {
        "select_terrain": EventTermCfg(func=env_mdp.randomize_terrain, mode="reset"),
        **env_cfg.events,
    }
    is_parkour = cfg.task in ("Unitree-Go2-PIE-Parkour", "Unitree-Go2-PIE-Parkour-AMP", "Unitree-Go2-MTE-Parkour-AMP")
    if cfg.speed is not None and not is_parkour:
        raise ValueError("--speed requires the parkour route task; use --command for legacy PIE.")
    direct_control = cfg.keyboard or cfg.command is not None
    env_cfg.commands["twist"].resample_on_reset_only = True
    if is_parkour:
        env_cfg.commands["twist"].speed_range = cfg.speed_range
    ranges = env_cfg.commands["twist"].ranges
    limits = np.asarray([ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z], dtype=np.float32)
    if direct_control and not is_parkour:
        _keyboard_command_limits(env_cfg, "twist")
    env = env_cfg.class_type(cfg=env_cfg, device=cfg.device, render_mode="rgb_array" if cfg.video else None)
    window = None
    try:
        if cfg.video:
            env = VideoRecorder(
                env, video_folder=checkpoint.parent / "videos" / "play_pie",
                step_trigger=lambda step: step == 0, video_length=cfg.video_length,
                disable_logger=True,
            )
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        route = vec.unwrapped.command_manager.get_term("twist")
        if isinstance(route, RouteVelocityCommand) and cfg.speed is not None:
            route.speed_override = cfg.speed
        runner = load_runner_cls(cfg.task)(vec, asdict(agent_cfg), device=cfg.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
        policy = runner.get_inference_policy(device=cfg.device)
        if cfg.export:
            runner.export_policy_to_onnx(str(checkpoint.parent), filename="policy.onnx")
            attach_metadata_to_onnx(str(checkpoint.parent / "policy.onnx"), runner.get_policy_metadata(str(checkpoint.parent)))
        if cfg.keyboard or cfg.depth:
            window = KeyboardController(limits if cfg.keyboard else None, depth=cfg.depth)
        controller = FixedCommandController(cfg.command) if cfg.command is not None else window if cfg.keyboard else None
        playback = PIEPlaybackEnv(vec, controller, "twist", policy, depth=cfg.depth, depth_window=window)
        # Reset once more after configuring speed overrides and recurrent state.
        playback.reset()
        mode = "keyboard" if cfg.keyboard else "direct velocity" if cfg.command is not None else "route" if is_parkour else "random velocity"
        print(f"[INFO] PIE playback: {mode}; terrain={cfg.terrain}; difficulty={cfg.terrain_level}/9")
        print(f"[INFO] Terrain length={cfg.terrain_length:g} m; gap={cfg.terrain_gap:g} m")
        segments = {name: terrain.repetitions for name, terrain in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()}
        print(f"[INFO] Obstacle segments per lane: {segments}")
        if is_parkour and not direct_control:
            speed_mode = f"fixed {cfg.speed:g}" if cfg.speed is not None else f"uniform {cfg.speed_range}, resampled only on reset"
            print(f"[INFO] Route speed: {speed_mode} m/s")
        if cfg.headless_steps:
            with torch.inference_mode():
                for _ in range(cfg.headless_steps):
                    actions = policy(playback.get_observations())
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Policy produced non-finite actions.")
                    playback.step(actions)
        elif _resolve_viewer(cfg.viewer) == "native":
            NativeMujocoViewer(playback, policy).run()
        else:
            ViserPlayViewer(playback, policy).run()
        report = {
            "task": cfg.task, "terrain": cfg.terrain, "terrain_level": cfg.terrain_level,
            "command_mode": mode,
            "terrain_length": cfg.terrain_length, "terrain_gap": cfg.terrain_gap,
            "terrain_segments": segments,
            "speed_range": cfg.speed_range, "speed_override": cfg.speed,
            "route_success_valid": is_parkour and not direct_control,
            "completed_episodes": len(playback.episodes),
            "route_successes": sum(bool(e.get("route_success", False)) for e in playback.episodes),
            "episodes": playback.episodes,
        }
        print(f"[INFO] Completed episodes: {report['completed_episodes']}; route successes: {report['route_successes']}")
        if cfg.stats_file:
            output = Path(cfg.stats_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if window is not None:
            window.close()
        env.close()


if __name__ == "__main__":
    run_play_pie(tyro.cli(PlayPIEConfig))

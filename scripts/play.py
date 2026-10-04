"""Play a velocity-controlled AMP policy with pygame keyboard commands."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# Prefer the repository's RSL-RL implementation over an installed package with
# the same import name when this script is executed as ``python scripts/play_amp.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


_PYGAME: Any | None = None


def _load_pygame() -> Any:
    """Import pygame only when keyboard playback is requested."""
    global _PYGAME
    if _PYGAME is None:
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Keyboard playback requires pygame. Install it with `pip install pygame`."
            ) from exc
        _PYGAME = pygame
    return _PYGAME


def get_keyboard_command(
    keys: Any,
    cmd_limits: np.ndarray,
    pygame_module: Any | None = None,
) -> np.ndarray:
    """Convert the W/S, A/D and Q/E key states to a velocity command.

    The command layout is ``[linear_x, linear_y, angular_z]``.  Each key uses
    the corresponding lower or upper limit, matching ``deploy_go2.py``.
    """
    pygame = pygame_module if pygame_module is not None else _load_pygame()

    if cmd_limits.shape != (3, 2):
        raise ValueError(f"Expected command limits with shape (3, 2), got {cmd_limits.shape}.")

    cmd_x = 0.0
    cmd_y = 0.0
    cmd_yaw = 0.0
    if keys[pygame.K_w]:
        cmd_x += cmd_limits[0, 1]
    if keys[pygame.K_s]:
        cmd_x += cmd_limits[0, 0]
    if keys[pygame.K_a]:
        cmd_y += cmd_limits[1, 1]
    if keys[pygame.K_d]:
        cmd_y += cmd_limits[1, 0]
    if keys[pygame.K_q]:
        cmd_yaw += cmd_limits[2, 1]
    if keys[pygame.K_e]:
        cmd_yaw += cmd_limits[2, 0]
    # cmd_x = 2.0
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


class KeyboardController:
    """Poll pygame and expose the current keyboard velocity command."""

    def __init__(self, cmd_limits: np.ndarray) -> None:
        self._pygame = _load_pygame()
        self._cmd_limits = cmd_limits
        self._pygame.init()
        self._screen = self._pygame.display.set_mode((200, 100))
        self._pygame.display.set_caption("AMP Keyboard Control")
        print("Keyboard control: W/S=vx, A/D=vy, Q/E=yaw")

    def read(self) -> np.ndarray:
        self._pygame.event.pump()
        keys = self._pygame.key.get_pressed()
        return get_keyboard_command(keys, self._cmd_limits, self._pygame)

    def close(self) -> None:
        self._pygame.quit()


class KeyboardCommandEnv:
    """Inject the keyboard command before each policy observation."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        keyboard: KeyboardController,
        command_name: str,
    ) -> None:
        self._env = env
        self._keyboard = keyboard
        self._command = env.unwrapped.command_manager.get_command(command_name)

        if self._command is None:
            raise ValueError(f"Command '{command_name}' is not available in the environment.")
        if self._command.ndim != 2 or self._command.shape[1] != 3:
            raise ValueError(
                f"Keyboard control expects a 3D velocity command, got {tuple(self._command.shape)}."
            )

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    def _update_command(self) -> None:
        command = torch.as_tensor(
            self._keyboard.read(), device=self._command.device, dtype=self._command.dtype
        )
        self._command.copy_(command.unsqueeze(0).expand_as(self._command))

    def initialize(self) -> None:
        """Replace the reset-time random command in the observation cache."""
        self._update_command()
        self._env.unwrapped.observation_manager.compute(update_history=True)

    def get_observations(self) -> Any:
        self._update_command()
        return self._env.get_observations()

    def step(self, actions: torch.Tensor) -> Any:
        return self._env.step(actions)

    def reset(self) -> Any:
        result = self._env.reset()
        self.initialize()
        return result

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


@dataclass(frozen=True)
class PlayConfig:
    """Configuration for keyboard-controlled AMP playback."""

    checkpoint_file: str | None = None
    command_name: str = "twist"
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = True


def _keyboard_command_limits(env_cfg: Any, command_name: str) -> np.ndarray:
    command_cfg = env_cfg.commands.get(command_name)
    if not isinstance(command_cfg, UniformVelocityCommandCfg):
        raise ValueError(
            f"Command '{command_name}' must be UniformVelocityCommandCfg for keyboard control."
        )

    # Disable all automatic command changes.  The command is updated by
    # KeyboardCommandEnv immediately before observations are computed.
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None

    ranges = command_cfg.ranges
    return np.asarray(
        [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z], dtype=np.float32
    )


def _resolve_viewer(viewer: str) -> Literal["native", "viser"]:
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return "native" if has_display else "viser"
    if viewer == "native":
        return "native"
    if viewer == "viser":
        return "viser"
    raise ValueError(f"Unsupported viewer backend: {viewer}")


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()

    if cfg.checkpoint_file is None:
        raise ValueError("AMP playback requires --checkpoint-file.")
    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled")
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    cmd_limits = _keyboard_command_limits(env_cfg, cfg.command_name)
    render_mode = "rgb_array" if cfg.video else None
    env_cls = getattr(env_cfg, "class_type", ManagerBasedRlEnv)
    env: Any = env_cls(cfg=env_cfg, device=device, render_mode=render_mode)

    log_dir = checkpoint_path.parent
    if cfg.video:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play_amp",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    onnx_path = log_dir / "policy.onnx"
    runner.export_policy_to_onnx(str(log_dir), filename="policy.onnx")
    actor_obs_cfg = env_cfg.observations["actor"]
    metadata = get_base_metadata(vec_env.unwrapped, str(log_dir))
    metadata.update(
        {
            "observation_history_layout": (
                "term-major" if actor_obs_cfg.flatten_history_dim else "frame-major"
            ),
            "observation_history_length": str(actor_obs_cfg.history_length),
        }
    )
    attach_metadata_to_onnx(str(onnx_path), metadata)
    print(f"[INFO] Exported ONNX policy to {onnx_path}")

    keyboard = KeyboardController(cmd_limits)
    keyboard_env = KeyboardCommandEnv(vec_env, keyboard, cfg.command_name)
    keyboard_env.initialize()
    resolved_viewer = _resolve_viewer(cfg.viewer)

    try:
        if resolved_viewer == "native":
            NativeMujocoViewer(keyboard_env, policy).run()
        else:
            ViserPlayViewer(keyboard_env, policy).run()
    finally:
        keyboard.close()
        keyboard_env.close()


def main() -> None:
    # Import tasks to populate the registry before tyro validates task choices.
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()

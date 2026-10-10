# ruff: noqa: E402
"""Play Go2 policies with random or keyboard commands and optional depth views."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# Prefer this repository over other editable installations of the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


_PYGAME: Any | None = None


def _load_pygame() -> Any:
    """Import pygame only for keyboard control or depth visualization."""
    global _PYGAME
    if _PYGAME is None:
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Keyboard/depth playback requires pygame. Install it with `pip install pygame`."
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
    """Display depth and optionally poll keyboard velocity commands."""

    def __init__(self, cmd_limits: np.ndarray | None, depth: bool = False) -> None:
        self._pygame = _load_pygame()
        self._cmd_limits = cmd_limits
        self._pygame.init()
        self._screen = self._pygame.display.set_mode((1280, 380) if depth else (200, 100))
        self._pygame.display.set_caption(
            "PIE Depth" if cmd_limits is None else (
                "Go2 Keyboard Control / PIE Depth" if depth else "Go2 Keyboard Control"
            )
        )
        self._font = self._pygame.font.Font(None, 23) if depth else None
        if cmd_limits is not None:
            print("Keyboard control: W/S=vx, A/D=vy, Q/E=yaw")

    def read(self) -> np.ndarray:
        assert self._cmd_limits is not None
        self._pygame.event.pump()
        keys = self._pygame.key.get_pressed()
        return get_keyboard_command(keys, self._cmd_limits, self._pygame)

    def close(self) -> None:
        self._pygame.quit()

    def show_depth(
        self, raw: np.ndarray, history: np.ndarray, cutoff: float
    ) -> None:
        """Display sensor metres and the exact cached policy frames for env 0."""
        assert self._font is not None
        pygame = self._pygame
        pygame.event.pump()
        self._screen.fill((24, 24, 28))

        def label(text: str, x: int, y: int) -> None:
            self._screen.blit(self._font.render(text, True, (235, 235, 235)), (x, y))

        label(
            "Environment 0 | W/S=vx, A/D=vy, Q/E=yaw (focus this window)"
            if self._cmd_limits is not None else "Environment 0 | PIE depth visualization",
            16, 10,
        )
        panels = (
            ("Raw depth (m), current sensor", raw, cutoff),
            ("Policy depth [0, 1], oldest", history[0], 1.0),
            ("Policy depth [0, 1], newest", history[-1], 1.0),
        )
        for index, (title, values, maximum) in enumerate(panels):
            x = 16 + index * 424
            label(title, x, 42)
            # Fixed display scale: never normalize each frame independently.
            # Raw invalid pixels remain distinguishable from far-range returns.
            valid = np.isfinite(values) & (values > 0)
            scaled = np.clip(np.where(valid, values, 0.0) / maximum, 0.0, 1.0)
            rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)
            rgb[~valid] = (255, 0, 255)
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            h, w = values.shape
            zoom = min(408 / w, 240 / h)
            surface = pygame.transform.scale(surface, (int(w * zoom), int(h * zoom)))
            self._screen.blit(surface, (x, 72))
            limits = (
                f"min={values[valid].min():.3f}, max={values[valid].max():.3f}"
                if valid.any() else "No valid depth"
            )
            label(f"{w} x {h} | {limits}", x, 318)
        label(
            f"Black=near, white=>={cutoff:g} m; magenta=invalid. "
            "Policy frames hold between camera updates; CNN subtracts 0.5.",
            16, 350,
        )
        pygame.display.flip()


class FixedCommandController:
    """A constant velocity command explicitly requested with --command."""

    def __init__(self, command: tuple[float, float, float]) -> None:
        self._command = np.asarray(command, dtype=np.float32)

    def read(self) -> np.ndarray:
        return self._command

    def close(self) -> None:
        pass


class KeyboardCommandEnv:
    """Handle optional command overrides, depth views and policy resets."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        keyboard: KeyboardController | FixedCommandController | None,
        command_name: str,
        policy: Any = None,
        depth: bool = False,
        depth_window: KeyboardController | None = None,
    ) -> None:
        self._env = env
        self._keyboard = keyboard
        self._command_name = command_name
        self._policy = policy
        self._depth = depth
        self._depth_window = depth_window or (
            keyboard if isinstance(keyboard, KeyboardController) else None
        )
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
        if self._keyboard is None:
            return
        command = torch.as_tensor(
            self._keyboard.read(), device=self._command.device, dtype=self._command.dtype
        )
        self._command.copy_(command.unsqueeze(0).expand_as(self._command))
        # mjlab caches observations. Replace only the current command and the
        # newest command-history sample; do not advance histories or resample
        # sensor noise when a viewer asks for observations a second time.
        manager = self._env.unwrapped.observation_manager
        observations = manager.compute()
        for group, names in manager.active_terms.items():
            if not manager.group_obs_concatenate[group]:
                continue
            offset = 0
            for name, shape in zip(names, manager.group_obs_term_dim[group], strict=True):
                width = int(np.prod(shape))
                term = manager.get_term_cfg(group, name)
                if term.params.get("command_name") == self._command_name and name == "command":
                    if not term.flatten_history_dim:
                        raise ValueError("Keyboard commands require flattened observation histories.")
                    value = self._command
                    if term.scale is not None:
                        value = value * term.scale
                    observations[group][:, offset + width - 3:offset + width] = value
                    if term.history_length > 0:
                        # mjlab 1.2 exposes buffers only through this mapping;
                        # CircularBuffer[0] is its writable newest frame.
                        history = manager._group_obs_term_history_buffer[group][name]
                        history[0].copy_(value)
                offset += width

    def initialize(self) -> None:
        """Apply an optional command override and initialize depth panels."""
        self._update_command()
        self._update_depth(self._env.get_observations())

    def _update_depth(self, observations: Any) -> None:
        if not self._depth:
            return
        assert self._depth_window is not None
        env = self._env.unwrapped
        params = env.cfg.observations["camera"].terms["front_depth"].params
        raw = env.scene[params["sensor_name"]].data.depth
        assert raw is not None
        height = raw.shape[1]
        width = raw.shape[2] - params.get("crop_left", 0) - params.get("crop_right", 0)
        if params.get("resize") is not None:
            height, width = params["resize"]
        # Read the cached observation passed to the policy. Reprocessing the
        # latest sensor frame here would incorrectly bypass the 10 Hz history.
        history = observations["camera"][0].reshape(
            params["frame_history_length"], height, width
        )
        self._depth_window.show_depth(
            raw[0, ..., 0].detach().cpu().numpy(),
            history.detach().cpu().numpy(),
            params["cutoff_distance"],
        )

    def get_observations(self) -> Any:
        self._update_command()
        observations = self._env.get_observations()
        self._update_depth(observations)
        return observations

    def step(self, actions: torch.Tensor) -> Any:
        result = self._env.step(actions)
        self._update_depth(result[0])
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset(result[2])
        return result

    def reset(self) -> Any:
        result = self._env.reset()
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
        self.initialize()
        return result

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


@dataclass(frozen=True)
class PlayConfig:
    """Configuration for Go2 policy playback."""

    checkpoint_file: str | None = None
    command_name: str = "twist"
    num_envs: int | None = None
    device: str | None = None
    video: tyro.conf.FlagConversionOff[bool] = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    keyboard: bool = False
    """Use keyboard velocity commands instead of the environment's random commands."""
    depth: bool = False
    """Show PIE raw depth and both processed policy frames in a separate window."""
    no_terminations: tyro.conf.FlagConversionOff[bool] = True
    headless_steps: int = 0
    command: tuple[float, float, float] | None = None
    """Override random commands with a fixed vx/vy/yaw command; excludes --keyboard."""


def _keyboard_command_limits(env_cfg: Any, command_name: str) -> np.ndarray:
    command_cfg = env_cfg.commands.get(command_name)
    if command_cfg is None or not all(
        hasattr(command_cfg, field)
        for field in ("ranges", "heading_command", "rel_standing_envs")
    ):
        raise ValueError(
            f"Command '{command_name}' must provide velocity command ranges."
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
    if cfg.headless_steps < 0:
        raise ValueError("--headless-steps must be nonnegative.")
    if cfg.depth and cfg.headless_steps > 0:
        raise ValueError("--depth requires interactive playback; omit --headless-steps.")
    if cfg.keyboard and cfg.headless_steps > 0:
        raise ValueError("--keyboard requires interactive playback; omit --headless-steps.")
    if cfg.keyboard and cfg.command is not None:
        raise ValueError("Use either --keyboard or --command, not both.")

    if cfg.checkpoint_file is None:
        raise ValueError("Playback requires --checkpoint-file.")
    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    if cfg.depth:
        camera_group = env_cfg.observations.get("camera")
        if camera_group is None or "front_depth" not in camera_group.terms:
            raise ValueError("--depth requires a PIE task with front_depth camera observations.")

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled")
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    cmd_limits = None
    if cfg.keyboard or cfg.command is not None:
        cmd_limits = _keyboard_command_limits(env_cfg, cfg.command_name)
    else:
        print("[INFO] Using environment random velocity commands")
    render_mode = "rgb_array" if cfg.video else None
    env_cls = getattr(env_cfg, "class_type", ManagerBasedRlEnv)
    env: Any = env_cls(cfg=env_cfg, device=device, render_mode=render_mode)

    log_dir = checkpoint_path.parent
    if cfg.video:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
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
    if hasattr(runner, "get_policy_metadata"):
        metadata = runner.get_policy_metadata(str(log_dir))
    else:
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

    window = (
        KeyboardController(cmd_limits if cfg.keyboard else None, depth=cfg.depth)
        if cfg.keyboard or cfg.depth else None
    )
    controller = (
        FixedCommandController(cfg.command) if cfg.command is not None else
        window if cfg.keyboard else None
    )
    keyboard_env = KeyboardCommandEnv(
        vec_env, controller, cfg.command_name, policy,
        depth=cfg.depth, depth_window=window,
    )
    resolved_viewer = _resolve_viewer(cfg.viewer)

    try:
        keyboard_env.initialize()
        if cfg.headless_steps > 0:
            with torch.inference_mode():
                for _ in range(cfg.headless_steps):
                    actions = policy(keyboard_env.get_observations())
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Policy produced non-finite actions.")
                    keyboard_env.step(actions)
            print(f"[INFO] Completed {cfg.headless_steps} headless steps")
        elif resolved_viewer == "native":
            NativeMujocoViewer(keyboard_env, policy).run()
        else:
            ViserPlayViewer(keyboard_env, policy).run()
    finally:
        if window is not None:
            window.close()
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
        # Keep existing explicit booleans; --keyboard and --depth are bare switches.
        config=tuple(flag for flag in mjlab.TYRO_FLAGS if flag is not tyro.conf.FlagConversionOff),
    )
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()

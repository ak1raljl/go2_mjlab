"""Motion playback script for validating Go2 NPZ files.

Reads the NPZ motion files produced by ``scripts/mocap_json2npz_go2.py`` (or by
``scripts/csv2npz.py`` for other robots) and plays them back on a mjlab Go2.

    python scripts/play_motion_go2.py --motion-file src/assets/motions/go2/mocap_all_lb/walk_0.npz
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig

from src.assets.robots import get_go2_robot_cfg


@dataclass
class RobotConfig:
    joint_names: list[str]

ROBOT_CONFIGS = {
    "go2": RobotConfig(
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
    ),
}

ROBOT_CFG_FACTORIES = {
    "go2": get_go2_robot_cfg,
}

DEFAULT_MOTION_DIR = Path("src/assets/motions/go2")


@dataclass(frozen=True)
class PlayConfig:
    motion_file: str
    robot: str = "go2"
    loop: bool = True
    viewer: Literal["auto", "native", "viser"] = "auto"
    device: str | None = None
    physics_dt: float | None = None
    follow: bool = True
    camera_distance: float = 1.6
    camera_elevation: float = -15.0
    camera_azimuth: float = 120.0


def build_viewer_config(cfg: PlayConfig) -> ViewerConfig:
    """Build the viewer camera config.

    The default ``ViewerConfig`` uses ``OriginType.AUTO``, which the offscreen
    renderer resolves to a *free* camera ("Free camera, no tracking." in
    ``mjlab/viewer/offscreen_renderer.py``) and the Viser viewer ignores
    entirely. These motion clips travel several metres, so that leaves the robot
    walking out of frame. Tracking the asset root instead keeps it centred.
    """
    return ViewerConfig(
        origin_type=(
            ViewerConfig.OriginType.ASSET_ROOT
            if cfg.follow
            else ViewerConfig.OriginType.WORLD
        ),
        entity_name="robot",
        distance=cfg.camera_distance,
        elevation=cfg.camera_elevation,
        azimuth=cfg.camera_azimuth,
    )


class NPZMotionLoader:
    """Loads and manages motion data from NPZ files."""

    def __init__(self, motion_file: str, device: str, loop: bool = True):
        self.motion_file = motion_file
        self.device = device
        self.loop = loop
        self.current_frame = 0

        data = np.load(motion_file)

        self.fps = float(data["fps"][0])
        self.joint_pos = torch.from_numpy(data["joint_pos"]).to(device).float()
        self.joint_vel = torch.from_numpy(data["joint_vel"]).to(device).float()
        self.body_pos_w = torch.from_numpy(data["body_pos_w"]).to(device).float()
        self.body_quat_w = torch.from_numpy(data["body_quat_w"]).to(device).float()

        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.num_frames / self.fps

        print(
            f"[INFO] Loaded motion: {self.num_frames} frames @ {self.fps:g} Hz "
            f"({self.duration:.2f}s)"
        )

    def get_frame(self, frame_idx: int | None = None):
        if frame_idx is None:
            frame_idx = self.current_frame

        root_pos = self.body_pos_w[frame_idx, 0]  # (3,)
        root_quat = self.body_quat_w[frame_idx, 0]  # (4,) in wxyz
        joint_pos = self.joint_pos[frame_idx]  # (num_joints,)
        joint_vel = self.joint_vel[frame_idx]  # (num_joints,)

        return root_pos, root_quat, joint_pos, joint_vel

    def step(self) -> bool:
        self.current_frame += 1

        if self.current_frame >= self.num_frames:
            if self.loop:
                self.current_frame = 0
                return False
            self.current_frame = self.num_frames - 1
            return True  # Signal end

        return False

    def reset(self):
        self.current_frame = 0


class MotionPlaybackEnv:
    """Lightweight environment for motion playback that mimics the RL env interface."""

    def __init__(
        self,
        scene: Scene,
        sim: Simulation,
        robot: Entity,
        motion: NPZMotionLoader,
        robot_joint_indices: torch.Tensor,
        viewer_cfg: ViewerConfig,
        physics_dt: float | None = None,
    ):
        self.scene = scene
        self.sim = sim
        self.robot = robot
        self.motion = motion
        self.robot_joint_indices = robot_joint_indices
        self.paused = False
        self.single_step_mode = False

        self.step_dt = 1.0 / motion.fps
        self.frame_dt = 1.0 / motion.fps
        self.physics_dt = physics_dt or self.sim.mj_model.opt.timestep

        self.num_envs = 1
        self.unwrapped = self

        from dataclasses import dataclass

        @dataclass
        class MinimalCfg:
            viewer: ViewerConfig

        self.cfg = MinimalCfg(viewer=viewer_cfg)

        class DummyRewardManager:
            def get_active_iterable_terms(self, env_idx=None):
                return []

        self.reward_manager = DummyRewardManager()

    def step(self, action=None):
        """Execute one simulation step with motion data."""
        if self.paused and not self.single_step_mode:
            return None, 0.0, False, False, {}

        self.single_step_mode = False

        should_end = self.motion.step()

        root_pos, root_quat, joint_pos, joint_vel = self.motion.get_frame()

        root_states = self.robot.data.default_root_state.clone()
        root_states[:, 0:3] = root_pos.unsqueeze(0)
        root_states[:, :2] += self.scene.env_origins[:, :2]
        root_states[:, 3:7] = root_quat.unsqueeze(0)
        self.robot.write_root_state_to_sim(root_states)

        joint_pos_full = self.robot.data.default_joint_pos.clone()
        joint_vel_full = self.robot.data.default_joint_vel.clone()
        joint_pos_full[:, self.robot_joint_indices] = joint_pos.unsqueeze(0)
        joint_vel_full[:, self.robot_joint_indices] = joint_vel.unsqueeze(0)
        self.robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

        self.sim.forward()
        self.scene.update(self.frame_dt)
        return None, 0.0, should_end, False, {}

    def reset(self, seed=None, options=None):
        self.motion.reset()
        self.scene.reset()
        return None, {}

    def toggle_pause(self):
        self.paused = not self.paused

    def single_step(self):
        if self.paused:
            self.single_step_mode = True

    def get_observations(self):
        return None

    def close(self):
        pass


def resolve_viewer(cfg: PlayConfig) -> str:
    """Resolve viewer type based on config and environment."""
    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        return "native" if has_display else "viser"
    return cfg.viewer


class DummyPolicy:
    def __call__(self, obs):
        return None


def run_play_motion(cfg: PlayConfig):
    """Main execution function for motion playback."""
    configure_torch_backends()

    if cfg.robot not in ROBOT_CONFIGS:
        available = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(f"Unsupported robot: {cfg.robot}. Available: {available}")

    motion_path = Path(cfg.motion_file).expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(
            f"Motion file not found: {motion_path}\n"
            f"Convert the dataset first, e.g.:\n"
            f"  python scripts/mocap_json2npz_go2.py --limit 5"
        )

    robot_cfg = ROBOT_CONFIGS[cfg.robot]
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    motion = NPZMotionLoader(str(motion_path), device=device, loop=cfg.loop)

    physics_dt = cfg.physics_dt or 1.0 / (motion.fps * 4)

    scene = Scene(
        SceneCfg(
            num_envs=1,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": ROBOT_CFG_FACTORIES[cfg.robot]()},
        ),
        device=device,
    )
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = physics_dt
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot: Entity = scene["robot"]

    expected_joints = len(robot_cfg.joint_names)
    if motion.joint_pos.shape[1] != expected_joints:
        raise ValueError(
            f"NPZ has {motion.joint_pos.shape[1]} joints but {cfg.robot} expects "
            f"{expected_joints}. The motion was likely built for another robot."
        )

    robot_joint_indices = robot.find_joints(robot_cfg.joint_names, preserve_order=True)[0]
    print(
        f"[INFO] Playback: 1 frame per step at {motion.fps:g} Hz "
        f"(physics dt={physics_dt:.6f} s)"
    )

    env = MotionPlaybackEnv(
        scene,
        sim,
        robot,
        motion,
        robot_joint_indices,
        viewer_cfg=build_viewer_config(cfg),
        physics_dt=physics_dt,
    )

    env.reset()
    policy = DummyPolicy()

    viewer_type = resolve_viewer(cfg)
    try:
        if viewer_type == "native":
            viewer = NativeMujocoViewer(env, policy)
            viewer.run()
        elif viewer_type == "viser":
            print("Web viewer started at http://localhost:8080")
            print("Use the web interface to control playback")
            print("Press Ctrl+C to exit")
            viewer = ViserPlayViewer(env, policy)
            viewer.run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {viewer_type}")
    finally:
        env.close()
        print("[INFO] Viewer closed")


def main():
    cfg = tyro.cli(
        PlayConfig,
        description="Motion playback for Go2 NPZ files",
    )

    run_play_motion(cfg)


if __name__ == "__main__":
    main()

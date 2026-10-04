"""Go2 NPZ expert data implementing the rsl_rl_amp data-source interface."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src import SRC_PATH

from .amp_features import AMP_FEATURE_LAYOUT, multi_body_state, quaternion_slerp


JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint"
  for leg in ("FL", "FR", "RL", "RR")
  for joint in ("hip", "thigh", "calf")
)
# Explicit order of the 13 link arrays in this repository's Go2 NPZ files.
BODY_NAMES = ("base_link",) + tuple(
  f"{leg}_{body}"
  for leg in ("FL", "FR", "RL", "RR")
  for body in ("hip", "thigh", "calf")
)


@dataclass
class MotionCfg:
  motion_dir: str = str(SRC_PATH / "assets" / "motions" / "go2")
  exclude_patterns: tuple[str, ...] = ("go2_jump_*.npz",)
  body_names: tuple[str, ...] = BODY_NAMES
  anchor_name: str = "base_link"
  preload_transitions: bool = True
  num_preload_transitions: int = 2_000_000
  preload_batch_size: int = 16_384


class Go2MotionLoader:
  """Cache interpolated expert transitions following amp_go2's preload design.

  NPZ joint order is FL/FR/RL/RR, hip/thigh/calf. Root arrays use body index
  zero and wxyz quaternions. The control rate must match the expert FPS.
  """

  def __init__(self, cfg: MotionCfg, step_dt: float, device: str):
    if not np.isfinite(step_dt) or step_dt <= 0:
      raise ValueError("AMP control step_dt must be positive and finite")
    if cfg.preload_batch_size <= 0:
      raise ValueError("AMP preload_batch_size must be positive")
    if cfg.preload_transitions and cfg.num_preload_transitions <= 0:
      raise ValueError("AMP num_preload_transitions must be positive when preloading")
    self.step_dt = step_dt
    self.body_names = tuple(cfg.body_names)
    self.anchor_name = cfg.anchor_name
    if not self.body_names or len(set(self.body_names)) != len(self.body_names):
      raise ValueError("AMP body_names must be nonempty and contain no duplicates")
    missing = set((*self.body_names, self.anchor_name)) - set(BODY_NAMES)
    if missing:
      raise ValueError(f"AMP bodies missing from Go2 NPZ data: {sorted(missing)}")
    self.body_ids = [BODY_NAMES.index(name) for name in self.body_names]
    self.anchor_id = BODY_NAMES.index(self.anchor_name)
    directory = Path(cfg.motion_dir).expanduser()
    excluded = {p for pattern in cfg.exclude_patterns for p in directory.glob(pattern)}
    self.motion_files = tuple(p for p in sorted(directory.glob("*.npz")) if p not in excluded)
    if not self.motion_files:
      raise FileNotFoundError(f"No selected NPZ motions in {directory}")
    self.device = device
    states, body_states, positions, velocities, lengths, frame_rates = [], [], [], [], [], []
    required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w")
    for path in self.motion_files:
      with np.load(path, allow_pickle=False) as data:
        fps = np.asarray(data["fps"]).reshape(-1)
        if fps.size != 1 or not np.isfinite(fps[0]) or not np.isclose(fps[0] * step_dt, 1.0):
          raise ValueError(f"{path}: expert FPS must match control dt={step_dt}")
        arrays = {key: np.asarray(data[key]) for key in required}
      length = arrays["joint_pos"].shape[0]
      if length < 2:
        raise ValueError(f"{path}: at least two frames are required")
      for key, array in arrays.items():
        expected = ((length, 12) if key.startswith("joint_") else
                    (length, len(BODY_NAMES), 4 if key == "body_quat_w" else 3))
        if array.shape != expected or not np.isfinite(array).all():
          raise ValueError(f"{path}: invalid {key}, expected finite shape {expected}")
      tensors = {key: torch.as_tensor(value, device=device, dtype=torch.float32)
                 for key, value in arrays.items()}
      body_quat = tensors["body_quat_w"]
      norm = body_quat.norm(dim=-1, keepdim=True)
      if (norm < 1e-6).any():
        raise ValueError(f"{path}: invalid body quaternion")
      body_quat = body_quat / norm
      q, qd = tensors["joint_pos"], tensors["joint_vel"]
      states.append(multi_body_state(
        tensors["body_pos_w"][:, self.body_ids], body_quat[:, self.body_ids],
        tensors["body_lin_vel_w"][:, self.body_ids], tensors["body_ang_vel_w"][:, self.body_ids],
        tensors["body_pos_w"][:, self.anchor_id], body_quat[:, self.anchor_id],
      ))
      body_states.append(torch.cat((tensors["body_pos_w"], body_quat,
                                    tensors["body_lin_vel_w"], tensors["body_ang_vel_w"]), dim=-1))
      positions.append(q)
      velocities.append(qd)
      lengths.append(length)
      frame_rates.append(float(fps[0]))
    self.states = torch.cat(states)
    self.body_states = torch.cat(body_states)
    self.root_states = self.body_states[:, 0]
    self.joint_pos = torch.cat(positions)
    self.joint_vel = torch.cat(velocities)
    self.lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    self.offsets = torch.cat((self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1]))
    self.fps = torch.tensor(frame_rates, device=device, dtype=torch.float32)
    self.durations = (self.lengths - 1) / self.fps
    if (self.durations + 1e-6 < step_dt).any():
      raise ValueError("Every expert clip must contain a full control-step transition")
    self.preloaded_s = None
    self.preloaded_s_next = None
    print(f"[AMP] Loaded {len(self.motion_files)} motions, {len(self.states)} frames, "
          f"state dimension {self.observation_dim}")
    if cfg.preload_transitions:
      self._preload_transitions(cfg.num_preload_transitions, cfg.preload_batch_size)
    else:
      print("[AMP] Expert preloading disabled; using continuous-time sampling on demand")

  @property
  def observation_dim(self) -> int:
    return 15 * len(self.body_names)

  @property
  def observation_spec(self) -> dict:
    """Persist frame conventions and body ordering with AMP checkpoints."""
    return {"version": 1, "body_names": self.body_names, "anchor_name": self.anchor_name,
            "layout": AMP_FEATURE_LAYOUT, "dimension": self.observation_dim}

  def sample_indices(self, batch_size: int, transition: bool = True) -> torch.Tensor:
    """Sample original frame indices for inspection; training samples continuous times."""
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    clips = torch.randint(len(self.motion_files), (batch_size,), device=self.states.device)
    count = self.lengths[clips] - int(transition)
    frames = (torch.rand(batch_size, device=self.states.device) * count).long()
    return self.offsets[clips] + frames

  def sample_times(self, batch_size: int, transition: bool = True):
    """Sample clips equally, then uniform valid times without crossing clip ends."""
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    clips = torch.randint(len(self.motion_files), (batch_size,), device=self.states.device)
    end = self.durations[clips] - (self.step_dt if transition else 0.0)
    times = torch.rand(batch_size, device=self.states.device) * end.clamp_min(0.0)
    return clips, times

  def _frame_indices(self, clips, times):
    # Actual frame timestamps are i/fps, with duration (N-1)/fps.
    phase = (times * self.fps[clips]).clamp_min(0.0)
    phase = torch.minimum(phase, (self.lengths[clips] - 1).float())
    # Snap timestamp roundoff at original frame boundaries (a few float ULPs).
    nearest = phase.round()
    tolerance = 2 * torch.finfo(phase.dtype).eps * phase.abs().clamp_min(1.0)
    phase = torch.where((phase - nearest).abs() <= tolerance, nearest, phase)
    lower = phase.floor().long()
    upper = torch.minimum(lower + 1, self.lengths[clips] - 1)
    return self.offsets[clips] + lower, self.offsets[clips] + upper, phase - lower

  def _interpolate_bodies(self, lower, upper, fraction):
    first, second = self.body_states[lower], self.body_states[upper]
    blend = fraction[:, None, None]
    result = torch.lerp(first, second, blend)
    result[..., 3:7] = quaternion_slerp(first[..., 3:7], second[..., 3:7], blend)
    return result

  def get_amp_frame_at_time_batch(self, clips, times):
    """Interpolate world link states, then encode anchor poses and local velocities."""
    lower, upper, fraction = self._frame_indices(clips, times)
    bodies = self._interpolate_bodies(lower, upper, fraction)
    selected, anchor = bodies[:, self.body_ids], bodies[:, self.anchor_id]
    return multi_body_state(selected[..., :3], selected[..., 3:7],
                            selected[..., 7:10], selected[..., 10:13],
                            anchor[..., :3], anchor[..., 3:7])

  @torch.no_grad()
  def _preload_transitions(self, count: int, batch_size: int):
    gib = count * 2 * self.observation_dim * self.states.element_size() / 1024**3
    print(f"[AMP] Preloading {count} expert transitions ({self.observation_dim}D, {gib:.2f} GiB)")
    self.preloaded_s = self.states.new_empty(count, self.observation_dim)
    self.preloaded_s_next = torch.empty_like(self.preloaded_s)
    for start in range(0, count, batch_size):
      stop = min(start + batch_size, count)
      clips, times = self.sample_times(stop - start)
      self.preloaded_s[start:stop].copy_(self.get_amp_frame_at_time_batch(clips, times))
      self.preloaded_s_next[start:stop].copy_(
        self.get_amp_frame_at_time_batch(clips, times + self.step_dt),
      )
    print(f"[AMP] Finished preloading {count} expert transitions; cache shape "
          f"{tuple(self.preloaded_s.shape)} per state")

  def sample(self, batch_size: int, batch_index: int = 0):
    del batch_index
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    if self.preloaded_s is not None:
      indices = torch.randint(len(self.preloaded_s), (batch_size,), device=self.states.device)
      return self.preloaded_s[indices], self.preloaded_s_next[indices]
    clips, times = self.sample_times(batch_size)
    return (self.get_amp_frame_at_time_batch(clips, times),
            self.get_amp_frame_at_time_batch(clips, times + self.step_dt))

  def sample_reference(self, batch_size: int):
    clips, times = self.sample_times(batch_size, transition=False)
    lower, upper, fraction = self._frame_indices(clips, times)
    first, second = self.root_states[lower], self.root_states[upper]
    blend = fraction[:, None]
    root = torch.lerp(first, second, blend)
    root[:, 3:7] = quaternion_slerp(first[:, 3:7], second[:, 3:7], blend)
    return (root, torch.lerp(self.joint_pos[lower], self.joint_pos[upper], blend),
            torch.lerp(self.joint_vel[lower], self.joint_vel[upper], blend))


def expert_data_from_env(env, device: str):
  """Reuse the environment's loader for discriminator sampling and resets."""
  dataset = env.unwrapped.motion_dataset
  if str(dataset.states.device) != str(torch.device(device)):
    raise ValueError("AMP environment and algorithm must use the same device")
  return dataset

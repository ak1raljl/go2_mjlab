"""AMP environment preserving physical terminal states before auto-reset."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.amp_loco.mdp.observations import amp_state
from src.tasks.amp_loco.rl.motion_loader import Go2MotionLoader, JOINT_NAMES


class AmpLocoEnv(ManagerBasedRlEnv):
  def load_managers(self):
    robot = self.scene["robot"]
    _, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(names) != JOINT_NAMES or tuple(robot.joint_names) != JOINT_NAMES:
      raise ValueError("AMP policy actions and NPZ data require FL/FR/RL/RR joint order")
    self.motion_dataset = Go2MotionLoader(self.cfg.motion, self.step_dt, self.device)
    self.amp_body_ids, body_names = robot.find_bodies(
      self.motion_dataset.body_names, preserve_order=True,
    )
    if tuple(body_names) != self.motion_dataset.body_names:
      raise ValueError("AMP model bodies must match the configured NPZ body order")
    anchor_ids, anchor_names = robot.find_bodies(self.motion_dataset.anchor_name)
    if anchor_names != [self.motion_dataset.anchor_name]:
      raise ValueError("AMP anchor must resolve to exactly one configured body")
    self.amp_anchor_body_id = anchor_ids[0]
    self._capture_terminal = False
    self._terminal_amp_state = torch.zeros(
      self.num_envs, self.motion_dataset.observation_dim, device=self.device,
    )
    self._terminal_amp_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    super().load_managers()

  def step(self, action):
    self._terminal_amp_mask.zero_()
    self._capture_terminal = True
    try:
      result = super().step(action)
    finally:
      self._capture_terminal = False
    self.extras["terminal_amp_state"] = self._terminal_amp_state
    self.extras["terminal_amp_mask"] = self._terminal_amp_mask
    return result

  def _reset_idx(self, env_ids=None):
    if self._capture_terminal:
      # Refresh kinematics before reading multi-body link states, while qpos
      # and qvel still describe the terminal physical state.
      self.sim.forward()
      self._terminal_amp_state[env_ids] = amp_state(self)[env_ids]
      self._terminal_amp_mask[env_ids] = True
    super()._reset_idx(env_ids)

"""Shared expert/rollout encoding matching rsl_rl_amp's multi-body states."""

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)


AMP_FEATURE_LAYOUT = "body_pos_anchor,body_ori6d_anchor,body_lin_vel_local,body_ang_vel_local"


def quaternion_slerp(first: torch.Tensor, second: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
  """Batch shortest-arc SLERP for unit wxyz quaternions without mutating inputs.

  Fraction has a trailing singleton dimension and broadcasts over bodies.
  Near-identical rotations use normalized linear interpolation for stability.
  """
  dot = (first * second).sum(dim=-1, keepdim=True)
  second = torch.where(dot < 0.0, -second, second)
  dot = dot.abs().clamp(max=1.0)
  angle = torch.acos(dot)
  denominator = torch.sin(angle).clamp_min(1e-6)
  spherical = (torch.sin((1.0 - fraction) * angle) * first
               + torch.sin(fraction * angle) * second) / denominator
  linear = torch.lerp(first, second, fraction)
  result = torch.where(dot > 0.9995, linear, spherical)
  return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def multi_body_state(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode [batch, bodies] link states as feature-major 15D/body vectors.

  Poses are relative to the anchor; velocities use each body's own frame
  without subtracting anchor velocity. Quaternions are unit wxyz. The 6D
  orientation is the first two rotation-matrix columns flattened row-major,
  exactly as in thirdparty/rsl_rl_amp/utils/motion_loader.py.
  """
  anchor_pos = anchor_pos_w.unsqueeze(1).expand_as(body_pos_w)
  anchor_quat = anchor_quat_w.unsqueeze(1).expand_as(body_quat_w)
  position, orientation = subtract_frame_transforms(
    anchor_pos, anchor_quat, body_pos_w, body_quat_w,
  )
  rotation6d = matrix_from_quat(orientation)[..., :, :2]
  linear = quat_apply_inverse(body_quat_w, body_lin_vel_w)
  angular = quat_apply_inverse(body_quat_w, body_ang_vel_w)
  return torch.cat(tuple(value.flatten(start_dim=1)
                         for value in (position, rotation6d, linear, angular)), dim=-1)

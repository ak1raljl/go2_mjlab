"""Reference-state initialization, independent of the velocity task."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul


def reset_reference_state(env, env_ids):
  """Apply expert states while retaining the preceding reset's position and rotation."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  elif isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  probability = env.cfg.reference_init_probability
  if not 0.0 <= probability <= 1.0:
    raise ValueError("reference_init_probability must be in [0, 1]")
  selected = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
  if len(selected) == 0:
    return
  root, q, qd = env.motion_dataset.sample_reference(len(selected))
  robot = env.scene["robot"]
  default_root = robot.data.default_root_state
  assert default_root is not None
  # Read qpos directly: link poses are stale until the reset's forward pass.
  reset_pose = env.sim.data.qpos[selected[:, None], robot.indexing.free_joint_q_adr]
  rotation = quat_mul(reset_pose[:, 3:7], quat_conjugate(default_root[selected, 3:7]))
  root[:, :2] = reset_pose[:, :2]
  root[:, 2] += reset_pose[:, 2] - default_root[selected, 2] + env.cfg.reference_height_offset
  root[:, 3:7] = quat_mul(rotation, root[:, 3:7])
  root[:, 7:10] = quat_apply(rotation, root[:, 7:10])
  root[:, 10:13] = quat_apply(rotation, root[:, 10:13])
  robot.write_root_state_to_sim(root, env_ids=selected)
  robot.write_joint_state_to_sim(q, qd, joint_ids=env.amp_joint_ids, env_ids=selected)

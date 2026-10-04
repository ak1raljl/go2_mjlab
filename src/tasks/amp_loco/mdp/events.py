"""Reference-state initialization, independent of the velocity task."""

import torch


def reset_reference_state(env, env_ids):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  probability = env.cfg.reference_init_probability
  if not 0.0 <= probability <= 1.0:
    raise ValueError("reference_init_probability must be in [0, 1]")
  selected = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
  if len(selected) == 0:
    return
  root, q, qd = env.motion_dataset.sample_reference(len(selected))
  root[:, :2] = env.scene.env_origins[selected, :2]
  root[:, 2] += env.scene.env_origins[selected, 2] + env.cfg.reference_height_offset
  robot = env.scene["robot"]
  robot.write_root_state_to_sim(root, env_ids=selected)
  robot.write_joint_state_to_sim(q, qd, joint_ids=env.amp_joint_ids, env_ids=selected)

"""Policy joint observations and reference-compatible multi-body AMP states."""

from src.tasks.amp_loco.rl.amp_features import multi_body_state


def joint_pos_rel(env):
  data = env.scene["robot"].data
  return (data.joint_pos - data.default_joint_pos)[:, env.amp_joint_ids]


def joint_vel(env):
  return env.scene["robot"].data.joint_vel[:, env.amp_joint_ids]


def amp_state(env):
  """Uncorrupted 15D/body link states, using the expert loader's body order."""
  data = env.scene["robot"].data
  ids, anchor = env.amp_body_ids, env.amp_anchor_body_id
  return multi_body_state(
    data.body_link_pos_w[:, ids], data.body_link_quat_w[:, ids],
    data.body_link_lin_vel_w[:, ids], data.body_link_ang_vel_w[:, ids],
    data.body_link_pos_w[:, anchor], data.body_link_quat_w[:, anchor],
  )

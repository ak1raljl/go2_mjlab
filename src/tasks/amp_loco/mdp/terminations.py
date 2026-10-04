"""Contact termination for the AMP task."""


def illegal_contact(env, sensor_name: str, force_threshold: float = 1.0):
  force = env.scene[sensor_name].data.force
  return (force.norm(dim=-1) > force_threshold).any(dim=-1)

"""Step-based command curricula kept independent of the velocity task."""

from typing import TypedDict

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]


def commands_vel(env, env_ids, command_name: str, velocity_stages: list[VelocityStage]) -> dict[str, float]:
  """Apply the latest stage on reset, using the checkpointed simulation-step counter.

  Steps count control ticks per environment, independently of environment count.
  Already active commands finish their sampling interval; new samples use the
  updated ranges. Every stage specifies all axes, so restores are deterministic.
  """
  del env_ids
  if not velocity_stages or velocity_stages[0]["step"] != 0:
    raise ValueError("Velocity stages must start at step zero")
  previous_step = -1
  axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
  for stage in velocity_stages:
    step = stage["step"]
    if not isinstance(step, int) or step <= previous_step:
      raise ValueError("Velocity stage steps must be strictly increasing integers")
    previous_step = step
    for axis in axes:
      lower, upper = stage[axis]
      if not (float("-inf") < lower <= upper < float("inf")):
        raise ValueError(f"Velocity stage {step}: invalid finite {axis} range")

  command = env.command_manager.get_term(command_name)
  if not isinstance(command.cfg, UniformVelocityCommandCfg):
    raise TypeError("Velocity command curriculum requires UniformVelocityCommandCfg")
  index = max(i for i, stage in enumerate(velocity_stages)
              if env.common_step_counter >= stage["step"])
  stage = velocity_stages[index]
  state = {"stage": float(index)}
  for axis in axes:
    lower, upper = stage[axis]
    setattr(command.cfg.ranges, axis, (lower, upper))
    state[f"{axis}_min"] = float(lower)
    state[f"{axis}_max"] = float(upper)
  return state

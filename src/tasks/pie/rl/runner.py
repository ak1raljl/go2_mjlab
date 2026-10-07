"""PIE checkpoints and the multi-input recurrent deployment contract."""

import json
from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class PIEOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def get_policy_metadata(self, run_path: str = "local") -> dict:
    env = self.env.unwrapped
    actor = self.alg.actor
    metadata = get_base_metadata(env, run_path)
    depth = env.cfg.observations["camera"].terms["front_depth"].params
    camera = next(s for s in env.cfg.scene.sensors if s.name == depth["sensor_name"])
    metadata.update({
      "policy_type": "pie",
      "pie_contract_version": "1",
      "observation_history_layout": "term-major",
      "observation_history_length": str(actor.history_length),
      "actor_history_length": "1",
      "proprio_history_order": "oldest-to-newest-within-each-term",
      "proprio_term_dimensions": json.dumps(
        env.observation_manager.group_obs_term_dim["actor"]
      ),
      "depth_history_shape": json.dumps(actor.depth_shape),
      "depth_history_order": "oldest-to-newest",
      "depth_raw_shape": json.dumps([camera.height, camera.width]),
      "depth_crop_left": str(depth["crop_left"]),
      "depth_crop_right": str(depth["crop_right"]),
      "depth_min_m": str(depth.get("min_depth", 0.05)),
      "depth_cutoff_m": str(depth["cutoff_distance"]),
      "depth_invalid_fill_m": str(depth["cutoff_distance"]),
      "depth_gaussian_blur": json.dumps(depth["gaussian_blur"]),
      "depth_normalization": "clamp(depth_m,min_m,cutoff_m)/cutoff_m",
      "depth_measurement": "range-along-unit-camera-ray-in-metres",
      "depth_centering": "subtract-0.5-inside-policy",
      "depth_update_period_steps": str(depth["update_period_steps"]),
      "policy_dt_s": str(env.step_dt),
      "camera_position": json.dumps(camera.pos),
      "camera_quaternion_wxyz": json.dumps(camera.quat),
      "camera_fovy_deg": str(camera.fovy),
      "camera_frame": "mujoco: optical -Z, image up +Y",
      "memory_shape": json.dumps([actor.memory_num_layers, 1, actor.memory_hidden_dim]),
      "memory_reset": "zeros-on-start-and-episode-reset",
      "history_reset": "repeat-first-frame",
      "latent_policy_mode": "mean",
      "policy_inputs": json.dumps({
        "proprio": [1, actor.proprio_dim],
        "proprio_history": [1, actor.history_dim],
        "depth_history": [1, *actor.depth_shape],
        "memory_h_in": [actor.memory_num_layers, 1, actor.memory_hidden_dim],
      }),
      "policy_outputs": json.dumps(["actions", "memory_h_out"]),
    })
    return metadata

  def save(self, path: str, infos=None):
    super().save(path, infos)
    directory = str(Path(path).parent)
    self.export_policy_to_onnx(directory, "policy.onnx")
    attach_metadata_to_onnx(
      str(Path(directory) / "policy.onnx"), self.get_policy_metadata(directory)
    )

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None or load_cfg.get("iteration", False):
      self.current_learning_iteration += 1
    return infos

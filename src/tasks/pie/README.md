# Go2 PIE

`Unitree-Go2-PIE` is an independent mjlab task adapted from the local
`thirdparty/pie` reference. It uses the existing Go2 model and actuators,
RSL-RL PPO, and a CNN/Transformer/GRU estimator. It has no AMP dependency.
Its terrain set is flat ground (10%), ascending stairs (45%) and descending
stairs (45%), with step heights from 0.03 to 0.25 m. This implementation does
not establish gap-jumping or other parkour capabilities.

## Train and resume

Use the existing `go2_mjlab` conda environment. Registration:

```bash
python scripts/list_envs.py --keyword Go2
```

Short validation run (checks execution, not locomotion proficiency):

```bash
python scripts/train.py Unitree-Go2-PIE --gpu-ids '[0]' \
  --env.scene.num-envs 32 --env.episode-length-s 1.0 \
  --agent.max-iterations 5 --agent.save-interval 2 \
  --agent.run-name pie_smoke --enable-nan-guard True
```

Training starts from scratch; velocity/AMP checkpoints have incompatible
observation and network shapes. For a full run:

```bash
python scripts/train.py Unitree-Go2-PIE --gpu-ids '[0]' \
  --env.scene.num-envs 128 --agent.run-name pie
```

Increase environments only after measuring memory and throughput. At 4096
environments and 24 rollout steps, float32 depth observations alone occupy
about 4.1 GB, before sequence padding, activations and simulator memory.
Depth history updates at 10 Hz, but mjlab 1.2 renders the sensor at each
50 Hz control step; the history interval does not reduce rendering work.

Logs use TensorBoard and `logs/rsl_rl/go2_pie/`. Checkpoints include optimizer
state, normalizers and iteration state. Each save exports `policy.onnx`.

```bash
python scripts/train.py Unitree-Go2-PIE --gpu-ids '[0]' \
  --env.scene.num-envs 128 --agent.resume True \
  --agent.load-run <run-directory-name> --agent.load-checkpoint model_1000.pt \
  --agent.max-iterations 1000 --agent.run-name pie_resume
```

For PIE, resume starts at the next iteration. `max-iterations` is the number
of additional iterations in this invocation. Episode-local histories and
GRU state restart with the new environments; this is not an exact simulator
trajectory continuation.

## Observation and action contract

| Group | Shape per environment | Use |
| --- | --- | --- |
| `actor` | 45 | Current proprioception |
| `proprio_history` | 450 | Ten samples, term-major |
| `camera` | 10320 | Two 60 x 86 depth frames, oldest first |
| `critic` | 246 | Clean proprioception, true velocity and 198 height samples |
| `velocity_target` | 3 | Auxiliary velocity supervision |
| `height_target` | 198 | Relative height scan divided by 5 m |
| `foot_clearance_target` | 4 | Local foot clearance divided by 0.6 m |
| `successor_target` | 45 | Clean next-step proprioception, assigned by PPO |
| `successor_valid` | 1 | Excludes resets/timeouts from successor supervision |

The current proprioceptive order is angular velocity (3), projected gravity
(3), command vx/vy/yaw (3), joint position relative to default (12), joint
velocity relative to default (12), previous action (12). There is no phase
or privileged height scan in the actor inputs.

History is `[angular_velocity_history, gravity_history, command_history,
joint_position_history, joint_velocity_history, action_history]`, with
oldest-to-newest samples within each term. It is **not** a flattening of ten
complete 45-element vectors. `PIEEnv` derives current proprioception from
the latest history samples, so noise agrees exactly between the two inputs.
Reset backfills only the affected environments with their first sample.

The camera renders 106 x 60 pixels, then crops ten columns from each side.
Nonpositive/nonfinite depths become 3 m, followed by a 3 x 3 Gaussian blur
(sigma 1, reflect padding), clipping to [0.05, 3] m and division by 3.
The model subtracts 0.5 internally; do not subtract it again in deployment.
The underlying MuJoCo-Warp output is distance along the unit camera ray.
An axial/Z-depth camera transport needs conversion to the same definition.

The nominal camera is at (0.345, 0, 0.07) relative to `base_link`, pitched
20 degrees down. Raw horizontal FOV is 87 degrees; the config converts
this to vertical FOV before cropping. Depth history advances every five
control steps and repeats the first frame after reset. Camera extrinsic/FOV
randomization is included in training.

Policy frequency is 50 Hz (0.005 s physics step, decimation 4). The policy
produces twelve offsets with `q_target = q_default + 0.25 * actions`, in
the exported joint order. Existing Go2 PD gains apply. PIE restricts thigh
joint upper limits to 2.2 rad in its own model configuration.

## Training changes from the reference

- The policy uses the latent mean during both rollout and PPO updates;
  reparameterized latent sampling remains in the auxiliary decoder. This
  avoids resampling a hidden policy input when recomputing action likelihoods.
- Successor loss excludes both terminated and timeout transitions because
  mjlab returns reset observations on those steps. Other auxiliary losses
  still use the current state's valid targets. Recurrent padding is masked.
- Current proprioception uses the same noisy sample as the history.
- Terrain configurations are deep-copied to keep play/train independent.
- Checkpoint export includes the multi-input contract and uses local logging
  by default. Training and playback attach the same PIE metadata.
- Playback updates only the newest command sample, never pushes a duplicate
  history frame, and resets the policy memory on automatic and manual reset.

Losses combine PPO and value/entropy terms with velocity, clearance,
height reconstruction and successor MSE, plus KL regularization (weight 4).
Only the auxiliary decoder samples the implicit latent. The height decoder
is training-only; actor input contains a 16-element terrain representation.

## Validate and play

CPU checks cover recurrent replay with asynchronous resets, action log
probabilities, masked auxiliary gradients and sequential ONNX parity:

```bash
python test/validate_pie.py
python test/validate_pie.py --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt
```

Keyboard playback and a bounded headless rollout:

```bash
python scripts/play.py Unitree-Go2-PIE \
  --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt
python scripts/play.py Unitree-Go2-PIE \
  --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt \
  --num-envs 4 --headless-steps 200 --command 0.5 0.0 0.0
```

The default playback disables terminations, as for the existing tasks.
Use `--no-terminations False` to exercise automatic resets. PIE's default
training commands have zero lateral velocity and nonnegative forward speed.

## Terrain evaluation and visual ablations

Evaluate trained checkpoints at fixed terrain levels with the same seed,
command and episode duration. The evaluator disables observation noise,
pushes and curriculum changes; startup domain randomization remains.

```bash
python test/evaluate_pie.py --checkpoint-file <checkpoint.pt> \
  --terrain-level 0 --depth-mode normal --output outputs/pie_normal.json
python test/evaluate_pie.py --checkpoint-file <checkpoint.pt> \
  --terrain-level 0 --depth-mode frozen --output outputs/pie_frozen.json
python test/evaluate_pie.py --checkpoint-file <checkpoint.pt> \
  --terrain-level 0 --depth-mode delayed --depth-delay-steps 5 \
  --output outputs/pie_delayed.json
```

Repeat for levels 0 through 9 and multiple seeds. Frozen input holds the
initial image pair for each episode. Delayed input adds control-step delay
to the normal 10 Hz history. Both reset their buffers at episode boundaries.
These are inference ablations; a separately trained blind baseline is still
needed for a fair learned-policy comparison.

JSON reports include completed episodes by terrain, failure and traversal
proxy rates, velocity/yaw RMSE, action changes and foot slip. The success
proxy requires surviving to episode end and finishing more than half a tile
from the origin, matching the curriculum distance criterion. It is not a
direct count of stairs climbed. Unfinished episodes are excluded from rates,
and terrain types with no completed episodes report null rates. Check videos
alongside these metrics before declaring locomotion success.

## ONNX deployment

| Input | Shape |
| --- | --- |
| `proprio` | [1, 45] |
| `proprio_history` | [1, 450] |
| `depth_history` | [1, 2, 60, 86] |
| `memory_h_in` | [1, 1, 128] |

Outputs are `actions` [1, 12] and `memory_h_out` [1, 1, 128]. Feed the latter
into the next call; initialize it to zero at startup/reset. Normalizers and
latent-mean inference are part of the exported network. ONNX batch size is
fixed at one. Metadata includes joint order, default pose, gains, action
scale, history ordering, camera/preprocessing parameters and recurrent state.

Actual hardware deployment additionally requires matching camera calibration,
depth convention, timestamps and actuator ordering, and measuring inference
latency. The present sensor model includes blur and calibration variation;
it does not yet model a particular real camera's dropout, latency or noise
distribution. Smoke checkpoints are not trained locomotion policies.

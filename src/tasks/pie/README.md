# Go2 PIE

`Unitree-Go2-PIE` is an independent mjlab task adapted from the local
`thirdparty/pie` reference. It uses the existing Go2 model and actuators,
RSL-RL PPO, and a CNN/Transformer/GRU estimator. It has no AMP dependency.
Its terrain set is flat ground (10%), ascending stairs (45%) and descending
stairs (45%), with step heights from 0.03 to 0.25 m. This implementation does
not establish gap-jumping or other parkour capabilities.

## PIE Parkour task

Preview all terrain types and difficulties without a checkpoint or GPU:

```bash
python scripts/view_pie_terrains.py
# Open http://127.0.0.1:8080 in a browser.
python scripts/view_pie_terrains.py --seed 123 --port 8081
# Generate/validate the full gallery without starting a server:
python scripts/view_pie_terrains.py --check
```

The interactive gallery contains nine terrain columns and ten difficulty
rows (0–9), generated directly from `parkour_terrains.py`. Select a terrain
and/or level to focus on it; toggle route markers, waypoint numbers and labels.
Green marks the spawn, orange marks the ordered route. Difficulty is exactly
`level / 9`, as in `play_pie.py`; training samples within difficulty bands.
The preview generates an independent random layout for each tile. Change
`--seed` to inspect different layouts at the same level; `--shared-layout`
reuses random draws across levels to isolate difficulty changes.
Restart the script after editing the terrain code. For a remote machine,
forward port 8080 over SSH or through your editor.

`Unitree-Go2-PIE-Parkour` adds a separate terrain/task configuration inspired by
the local `thirdparty/extreme-parkour` implementation. It retains PIE's network,
observation dimensions, depth processing and auxiliary losses. The policy
receives **body-frame `[vx, vy, yaw_rate]` commands directly**. An external
controller converts the next terrain waypoint and a randomly sampled speed
into these commands; no waypoint coordinates, heading labels or learned
direction-prediction branch are added to the actor.

Each 18 x 4 m tile supplies eight ordered support points. The training speed is sampled
from 0.3–1.2 m/s every six seconds, with lateral commands limited to ±0.35 m/s
and yaw rate to ±1.2 rad/s. A waypoint counts after reaching/crossing it with
foot support at the corresponding height. Completing all points ends the
episode; falling, torso collisions and leaving the tile terminate it. Training
raises terrain difficulty after successful traversal and lowers it after
failure or insufficient progress. Episodes last at most 40 seconds.

| Terrain | Weight | Difficulty range |
| --- | --- | --- |
| `flat` | 10% | Ground with stronger surface roughness |
| `hurdle` | 15% | Six randomly spaced hurdles, nominal height 0.05–0.75 m |
| `step` | 15% | Six terraces with random lengths, increments 0.03–0.35 m |
| `gap` | 15% | Six randomly spaced gaps, width 0.10–1.0 m |
| `platform` | 15% | Random 1–3 platforms, length 1.4–2.2 m, height 0.05–0.75 m |
| `stairs_up` | 10% | Eight smooth 0.30 m treads, rise 0.03–0.25 m |
| `stairs_down` | 10% | Smooth descending counterpart |
| `slope_up` | 5% | Uphill ramp, inclination 5–25 degrees, length 4–6 m |
| `slope_down` | 5% | Downhill counterpart |

Hurdle/gap spacing varies within each tile, including at fixed difficulty.
Placement preserves at least 1.1 m of clear ground between obstacles. Platform
count, length, position and lateral offset are sampled per tile, with at least
1.6 m between platforms. Every platform has a top waypoint and a landing
waypoint; shorter routes add ground waypoints to retain eight valid points.
Stair and slope starting positions also vary. Geometry is sampled when the
terrain map is generated, not regenerated on every environment reset.

The obstacle layout is built from rectangular sections. Except for smooth
stairs, sections are converted to solid MuJoCo heightfields with physical
surface roughness, including obstacle tops and gap bottoms. Slopes use
continuous height profiles with roughness added. Separate heightfields
preserve vertical boundaries and gap widths. Spawn/goal heights follow the
actual rough surface. These are adaptations, not identical Isaac Gym terrains.
Hurdle/platform heights also have per-obstacle variation.

Roughness amplitude is sampled once per tile: **0.04–0.10 m for flat**, and
**0.02–0.06 m for obstacles and slopes**, and **zero for both stair types**.
Heights vary on both sides of the nominal
surface; these ranges describe amplitude, not peak-to-peak height. Quantized
uniform noise uses 0.005 m vertical increments on a 0.075 m coarse grid and is
interpolated onto a grid with at most 0.05 m spacing. Roughness is active at
every difficulty, as in the reference's `add_roughness()` default. The parameters
live in `ParkourTerrainCfg` and the per-type overrides in `PIE_PARKOUR_TERRAINS_CFG`.
Set `roughness_height_range=(0.0, 0.0)` to compare with smooth terrain.
Training, playback and the gallery all use this same terrain generator.
The reward favors progress toward the route at the requested speed, retains
yaw tracking, and relaxes pitch/roll and vertical-velocity penalties on
obstacles. Fixed gait-phase and standing-height rewards are removed to allow
jumping. Training results must establish whether the harder obstacles are
actually traversable by the learned policy.

```bash
python scripts/train.py Unitree-Go2-PIE-Parkour --gpu-ids '[0]' \
  --env.scene.num-envs 128 --agent.run-name pie_parkour
```

Logs and checkpoints use `logs/rsl_rl/go2_pie_parkour/`. The original PIE task
remains available. Existing PIE checkpoints load because the policy contract
is unchanged; they need further training to acquire the new obstacle skills.

The dedicated playback entry is `scripts/play_pie.py`:

```bash
# Defaults to route guidance with a speed sampled once at each reset.
python scripts/play_pie.py --checkpoint-file <checkpoint.pt> --depth

# Direct user velocity control; no automatic route steering.
python scripts/play_pie.py --checkpoint-file <checkpoint.pt> --keyboard --depth

# Select a terrain/difficulty and use a fixed route speed.
python scripts/play_pie.py --checkpoint-file <checkpoint.pt> \
  --terrain gap --terrain-level 3 --speed 0.8 --depth

# Bounded playback with a direct body-frame velocity command.
python scripts/play_pie.py --checkpoint-file <checkpoint.pt> \
  --terrain flat --command 0.5 0.1 0.0 --num-envs 4 --headless-steps 200 \
  --stats-file outputs/pie_parkour_play.json

# Replay the original task using the same standalone entry.
python scripts/play_pie.py --task Unitree-Go2-PIE \
  --checkpoint-file <checkpoint.pt> --depth
```

`--terrain-level` fixes difficulty from 0 to 9. With `--terrain all`, resets
sample a terrain family uniformly. Terminations and recurrent-state resets
remain enabled. `--depth` displays raw sensor depth and the two cached input
frames actually supplied to the policy, using the training preprocessing.
`--keyboard`, `--command`, and `--speed` are mutually exclusive. Direct controls
bypass route advancement, so their statistics are not route success estimates.
Add `--export` to explicitly write ONNX beside the checkpoint.

Playback samples the route speed only on reset and holds it throughout the
episode. Body-frame velocity components and yaw rate still follow the current
route direction. Pushes and all startup randomization (friction, mass, COM,
encoder bias, gains, motor strength and camera) are disabled; joint resets
use the nominal pose and zero velocity. Only robot state reset and terrain
selection events remain. Training keeps periodic speed sampling and its
original randomization events.

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

Random-command playback with depth, optional keyboard control, and a bounded
headless rollout with a fixed command:

```bash
python scripts/play.py Unitree-Go2-PIE \
  --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt --depth
python scripts/play.py Unitree-Go2-PIE \
  --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt --keyboard --depth
python scripts/play.py Unitree-Go2-PIE \
  --checkpoint-file logs/rsl_rl/go2_pie/<run>/model_4.pt \
  --num-envs 4 --headless-steps 200 --command 0.5 0.0 0.0
```

The default playback disables terminations, as for the existing tasks.
Velocity commands are sampled once at each reset and held for the episode,
including headless playback. Add `--keyboard` to override
them with W/S, A/D and Q/E, or explicitly use `--command vx vy yaw` for a fixed
command. `--keyboard` requires interactive playback and excludes `--command`.

Add `--depth` for three panels in a separate window: environment 0's raw
106 x 60 sensor depth in metres, and both 86 x 60 policy input frames
(oldest/newest). Policy panels read the actual cached camera observation,
including training's crop, invalid-depth replacement, Gaussian blur, clipping,
normalization and 10 Hz history updates. The CNN's internal subtraction of
0.5 follows these displayed [0, 1] observations. Raw depth updates at the
sensor rate, so it can change while policy frames are held. All panels use
a fixed near-to-far grayscale (white at 3 m); raw invalid pixels are magenta.
Display clipping does not modify the sensor or policy inputs. With `--keyboard`,
focus this window for keyboard commands. `--depth` works independently of
`--keyboard` and requires interactive playback.

Use `--no-terminations False` to exercise automatic resets. PIE's default
training commands have zero lateral velocity and nonnegative forward speed.

## Terrain evaluation and visual ablations

Evaluate trained checkpoints at fixed terrain levels with the same seed,
command and episode duration. The evaluator disables observation noise,
pushes, startup domain randomization and curriculum changes.

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

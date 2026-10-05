# Go2 MJLab

## Installation

**Conda environment**

```bash
conda create -n go2_mjlab python=3.11 -y
conda activate go2_mjlab
```

**Install dependencies**
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

**Clone Repo**
```bash
git clone https://github.com/ak1raljl/go2_mjlab.git
```

```bash
cd go2_mjlab
pip install -e .
```

## AMP locomotion

`Unitree-Go2-AMP-Loco` extends `Unitree-Go2-Flat` with AMP training. Its AMP
design follows `thirdparty/rsl_rl_amp`: PPO auxiliary discriminator loss,
joint Adam parameter groups, replay sampling, and mixed reward logging.
The installed RSL-RL package and `thirdparty` sources are not modified.
`amp_loco_env_cfg.py` owns `make_amp_loco_env_cfg()` and the full task
configuration. Its Go2 adapter owns robot assets, sensors, per-robot terms,
flat terrain, expert data, and playback overrides. The task's local `mdp/`
contains its locomotion observations, rewards, terminations, and curricula.
These settings match `Unitree-Go2-Flat`; actor/critic networks and PPO
hyperparameters match velocity as well. There are no imports from
`src.tasks.velocity`, so removing that package leaves AMP registration and
training functional. Configuration factories create independent objects;
robot configuration is deep copied. `env.py` captures terminal AMP states
before automatic resets.
Keep aligned locomotion settings in both tasks in sync explicitly when
either task changes.

The default expert dataset is `src/assets/motions/go2`: 20 NPZ clips,
including canter and excluding `go2_jump_*.npz`, sampled with equal clip
weights at 50 Hz. AMP uses the `rsl_rl_amp` multi-body representation for
all 13 NPZ links (`base_link`, then FL/FR/RL/RR hip/thigh/calf). Each state
contains positions and 6D orientations relative to the `base_link` anchor,
followed by linear/angular velocities in each body's own local frame.
Features are flattened by type, with 15 dimensions per body (195 total);
the single, two-frame ReLU discriminator is `390 → 1024 → 512 → 1`.
Body selection and anchor are configurable through `--env.motion.body-names`
and `--env.motion.anchor-name`, with the same ordering used for expert and
simulation observations. No additional foot positions or base height are
appended to the discriminator input.
Actor/critic observations match flat velocity, including the actor's gait
phase and the critic's foot height, air time, contacts, and forces. Robot PD
gains, action offsets, action scale (0.25), and the 50 Hz control rate match
velocity. Earlier AMP checkpoints with 45/48 actor/critic inputs require a
new training run because these observation layouts changed.

### Expert transition preloading

The expert loader follows `thirdparty/amp_go2`'s continuous-time transition
preloading design. It chooses clips with equal probability, samples a valid
time `t`, and caches states at `t` and `t + step_dt`. Position and world
linear/angular velocity use linear interpolation; every body orientation
uses shortest-arc quaternion SLERP. Multi-body features are encoded after
interpolation, preserving unit rotations and the 195D frame layout.
Time indexing uses actual frame timestamps `i / fps`, with both states
inside the same clip.

The Go2 configuration currently preloads **1,000,000 expert transitions**,
filled in batches of 16,384. Two float32 `(1000000, 195)` tensors consume
about 1.56 GB (1.45 GiB), in addition to the policy replay buffer. Setting
2,000,000 transitions consumes 3.12 GB (2.91 GiB). The environment and
algorithm share one loader and one expert cache. Reference resets sample
continuous times from the original trajectories, including interpolated
root pose, velocity, and joint state. Playback disables the expert cache.

Configure training through `--env.motion.num-preload-transitions 2000000`
and `--env.motion.preload-batch-size 16384`. To sample continuous transitions
on demand, set `--env.motion.preload-transitions False`. Expert and policy
transition caches are rebuilt on resume. Startup logs first report the
original 20 clips / 3896 frames, then the expert preload count and shapes.

### Velocity command curriculum

The curriculum uses velocity's schedule and implementation at episode
resets, with `common_step_counter` counting control steps per environment.
Stage thresholds use `step > threshold`, matching velocity exactly:

| Control step | vx (m/s) | vy (m/s) | wz (rad/s) |
|---:|---|---|---|
| > 0 | [-0.5, 1.0] | [-0.5, 0.5] | [-1.0, 1.0] |
| > 120000 (5000 × 24) | [-1.0, 2.0] | [-1.0, 1.0] | [-1.0, 1.0] |

At control step zero, command ranges retain velocity's base configuration:
vx [-1, 2], vy [-1, 1], wz [-1, 1]. The aligned schedule is defined in
`src/tasks/amp_loco/amp_loco_env_cfg.py`; the flat task disables terrain
curriculum. Commands resample every 3–8 seconds with heading control enabled.
Checkpoints restore the step counter. Playback follows flat velocity:
curriculum and pushes are disabled; ranges are vx [-0.5, 1], vy [-0.5, 0.5],
wz [-0.5, 0.5].

Task rewards use all 15 velocity terms and their exact weights, including
posture, gait, clearance/slip, angular momentum, and termination penalty.
Terminations are timeout, 70-degree tilt, and non-foot contact exceeding
10 N, using the same four-frame contact history as velocity. AMP adds its
existing discriminator reward and reference initialization to this task.

Analyze the stored motions with `python scripts/analyze_motion_speeds.py`.
The [speed report](docs/motion_analysis/go2/README.md) includes every clip,
root-frame speed quantiles, and a clip-uniform raw-frame reference distribution;
CSV, JSON, and a plot are saved beside it. Linear velocities are in m/s and
angular velocities in rad/s. These are observed expert velocities, rather
than recorded target commands.

```bash
python scripts/list_envs.py --keyword Go2
python scripts/train.py Unitree-Go2-AMP-Loco --gpu-ids '[0]' \
  --env.scene.num-envs 32 --env.episode-length-s 1.0 \
  --agent.max-iterations 5 --agent.save-interval 2 \
  --agent.run-name amp_loco_smoke --enable-nan-guard True
```

Training logs, checkpoints, and `policy.onnx` are written to
`logs/rsl_rl/go2_amp_loco/<timestamp>_<run_name>/`. Checkpoints include
discriminator, AMP normalizer, and observation-layout state; replay is
rebuilt on resume. Full resumes require identical body ordering, anchor,
and feature layout. Earlier 30D AMP checkpoints require a new training run
or explicitly loading only actor/critic weights.
The exported ONNX policy includes actor normalization and observation
metadata. Reference initialization is configurable through
`--env.reference-init-probability`; the default is 1.0 for training and
0.0 for playback. A configurable 1 cm reset height offset compensates for
the NPZ foot clearance relative to the current MJCF foot sphere.

Training resets randomize XY by ±0.5 m and yaw by ±3.14 rad. Reference
initialization retains these transforms and rotates expert world velocities
consistently. Startup events randomize foot friction in [0.3, 1.6] (shared
across the four feet per environment), encoder bias by ±0.015 rad, and base
COM by ±0.05 m on each axis. Encoder bias is applied to joint position
targets following velocity; policy observations and AMP use physical
states. Random velocity pushes occur every 5–6 seconds and are disabled
during playback.

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
cd parkour_mjlab
pip install -e .
```

## AMP locomotion

`Unitree-Go2-AMP-Loco` is a flat-terrain task implemented in
`src/tasks/amp_loco`, independently of the velocity tasks. Its AMP training
design follows `thirdparty/rsl_rl_amp`: PPO auxiliary discriminator loss,
joint Adam parameter groups, replay sampling, and mixed reward logging.
The installed RSL-RL package and `thirdparty` sources are not modified.
Environment configuration follows the velocity task structure:
`amp_loco_env_cfg.py` provides `make_amp_loco_env_cfg()` for common task
settings, and `config/go2/env_cfgs.py` fills robot assets, contact sensors,
motion data, and playback overrides. `env.py` handles runtime AMP state
capture before automatic resets.

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
Actor/critic observations have 45/48 dimensions. Robot PD gains, default
action offsets, action scale (0.25), and the 50 Hz control rate match the
velocity tasks. The forward command range extends to 3 m/s for canter.

### Expert transition preloading

The expert loader follows `thirdparty/amp_go2`'s continuous-time transition
preloading design. It chooses clips with equal probability, samples a valid
time `t`, and caches states at `t` and `t + step_dt`. Position and world
linear/angular velocity use linear interpolation; every body orientation
uses shortest-arc quaternion SLERP. Multi-body features are encoded after
interpolation, preserving unit rotations and the 195D frame layout.
Time indexing uses actual frame timestamps `i / fps`, with both states
inside the same clip.

Training defaults to **2,000,000 expert transitions**, filled in batches of
16,384. The two float32 `(2000000, 195)` tensors consume about 3.12 GB
(2.91 GiB), in addition to the policy replay buffer. The environment and
algorithm share one loader and one expert cache. Reference resets sample
continuous times from the original trajectories, including interpolated
root pose, velocity, and joint state. Playback disables the expert cache.

Configure training through `--env.motion.num-preload-transitions 2000000`
and `--env.motion.preload-batch-size 16384`. To sample continuous transitions
on demand, set `--env.motion.preload-transitions False`. Expert and policy
transition caches are rebuilt on resume. Startup logs first report the
original 20 clips / 3896 frames, then the expert preload count and shapes.

### Velocity command curriculum

Training expands the command ranges at episode resets using
`common_step_counter` (control steps per environment). With the default
24-step PPO rollout, the stages correspond to iterations 0/1000/3000/5000:

| Control step | vx (m/s) | vy (m/s) | wz (rad/s) |
|---:|---|---|---|
| 0 | [-0.5, 1.0] | [-0.3, 0.3] | [-0.5, 0.5] |
| 24000 | [-0.8, 1.5] | [-0.5, 0.5] | [-0.75, 0.75] |
| 72000 | [-1.0, 2.0] | [-0.6, 0.6] | [-1.0, 1.0] |
| 120000 | [-1.2, 3.0] | [-0.8, 0.8] | [-1.0, 1.0] |

The schedule is configured in `config/go2/env_cfgs.py`; the independent AMP
curriculum term lives in `mdp/curriculums.py`. Already sampled commands keep
their targets until resampling. Checkpoints restore the counter, and the
appropriate stage is reapplied at the next reset. Playback disables the
curriculum and uses the final ranges. TensorBoard records the stage and all
axis limits under `Curriculum/command_vel/*`.

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

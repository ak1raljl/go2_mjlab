# Go2 MJLab

Unitree Go2 quadruped RL training built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo + Warp) and RSL-RL PPO.

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

## Tasks

| Task ID | Description |
| --- | --- |
| `Unitree-Go2-Flat` | Velocity tracking on flat terrain |
| `Unitree-Go2-Rough` | Velocity tracking on rough terrain |
| `Unitree-Go2-AMP-Loco` | Flat velocity tracking with AMP style rewards |
| `Unitree-Go2-AMP-Rough` | AMP locomotion on generated rough terrain with critic-only scan |

```bash
python scripts/list_envs.py --keyword Go2
```

## Train

```bash
python scripts/train.py Unitree-Go2-Flat  --gpu-ids 0 --agent.run-name flat
python scripts/train.py Unitree-Go2-Rough --gpu-ids 0 --agent.run-name rough
python scripts/train.py Unitree-Go2-AMP-Loco --gpu-ids 0 --agent.run-name amp
python scripts/train.py Unitree-Go2-AMP-Rough --gpu-ids '[0]' --env.scene.num-envs 1024 --agent.run-name amp_rough
```
<details>
<summary><b>Script arguments</b></summary>

| Argument | Default | Description |
| --- | --- | --- |
| `--gpu-ids` | `[0]` | GPU ids to use; `all` for every visible GPU, `None` for CPU. More than one id launches a torchrunx multi-GPU run. |
| `--video` | `False` | Record training videos (`MUJOCO_GL=egl` is set automatically). |
| `--video-length` | `200` | Length of each recorded video, in steps. |
| `--video-interval` | `2000` | Record a video every N steps. |
| `--enable-nan-guard` | `False` | Enable the NaN guard (dumps diagnostics instead of silently diverging). |
| `--motion-file` | `None` | Only used by tracking tasks; not needed for the Go2 tasks here. |
</details>

<details>
<summary><b>Commonly overridden config fields</b></summary>

| Argument | Default | Description |
| --- | --- | --- |
| `--env.scene.num-envs` | `1` | Number of parallel environments; raise it to match your GPU. |
| `--agent.run-name` | `""` | Label appended to the timestamped run directory. |
| `--agent.max-iterations` | `10001` | Total training iterations. |
| `--agent.save-interval` | `1000` | Save a checkpoint every N iterations. |
| `--agent.resume` | `False` | Resume from `--agent.load-run` / `--agent.load-checkpoint`. |
| `--agent.load-run` | `.*` | Run directory to resume from (regex, latest match wins). |
| `--agent.load-checkpoint` | `model_.*.pt` | Checkpoint file to resume from (regex, latest match wins). |
| `--agent.seed` | `42` | Random seed. |

Logs, checkpoints and the exported `policy.onnx` are written to
`logs/`
</details>

## Play

```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<step>.pt
```

Keyboard control: `W`/`S` forward/backward, `A`/`D` lateral, `Q`/`E` yaw.

## Other scripts

| Script | Description |
| --- | --- |
| `scripts/play_motion_go2.py --motion-file <npz>` | Replay a reference motion clip on Go2 |

## AMP rough terrain

`Unitree-Go2-AMP-Rough` is an independent registered task using
`src/tasks/amp_loco/config/go2/terrains.py`. It matches `amp_go2`'s terrain
weights and parameter ranges: 10% smooth slopes (split equally between the
two signs), 20% rough slopes with [-0.05, 0.05] m noise sampled every 0.2 m,
25% stairs up, 25% stairs down, and 20% discrete obstacles. Slopes range from
0 to 0.4 rise/run, stair heights from 0.05 to 0.23 m with 0.31 m treads, and
obstacle heights from 0.05 to 0.25 m. Each obstacle patch has 20 rectangles
with widths/lengths of 1 to 2 m. Patches are 8 x 8 m with 3 m center platforms,
on a 10 x 20 curriculum grid with a 25 m border. mjlab uses native box stairs
and heightfields, so geometry and row difficulty sampling are not an exact
reproduction of Isaac Gym's triangle mesh.

Only the critic receives the 187-point, yaw-aligned terrain scan over
1.6 x 1.0 m at 0.1 m resolution. Its values follow `amp_go2`:
`clip(base_z - ground_z - 0.5, -1, 1) * 2.5`. Rays include only terrain
geometry (group 0). The critic's four foot heights use local terrain
clearance rather than absolute world Z. Actor and AMP discriminator inputs
keep their existing layouts. Rough training uses terrain and velocity
curricula; playback disables curricula and pushes, and selects a terrain
patch before resetting the robot onto its origin.

```bash
python scripts/train.py Unitree-Go2-AMP-Rough --gpu-ids '[0]' \
  --env.scene.num-envs 1024 --agent.run-name amp_rough

# Short smoke run with a smaller expert cache.
python scripts/train.py Unitree-Go2-AMP-Rough --gpu-ids '[0]' \
  --env.scene.num-envs 32 --env.episode-length-s 1.0 \
  --env.motion.num-preload-transitions 4096 \
  --agent.max-iterations 5 --agent.save-interval 2 \
  --agent.run-name amp_rough_smoke --enable-nan-guard True
```

Rough logs, checkpoints, and ONNX exports go to `logs/rsl_rl/go2_amp_rough/`.
Full flat checkpoints cannot resume rough training because the critic gains
187 inputs; start a fresh rough run. Playback uses the same task ID:
`python scripts/play.py Unitree-Go2-AMP-Rough --checkpoint-file <model.pt>`.

Both AMP tasks use ordinary resets following `amp_go2`; reference-state
initialization and its CLI options have been removed. The default base
height is 0.42 m above the terrain origin, root orientation is the default
quaternion, root linear/angular velocities are sampled in [-0.5, 0.5],
and joint angles are scaled by [0.5, 1.5] from their defaults (clamped
to soft limits) with zero joint velocities. Rough resets randomize XY
by +/-1 m around the terrain origin; flat resets use the origin directly.

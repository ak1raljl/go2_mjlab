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
| `Unitree-Go2-PIE` | Recurrent depth-based locomotion on flat terrain and stairs |

```bash
python scripts/list_envs.py --keyword Go2
```

## Train

```bash
python scripts/train.py Unitree-Go2-Flat  --gpu-ids '[0]' --env.scene.num-envs 4096 --agent.run-name flat
python scripts/train.py Unitree-Go2-Rough --gpu-ids '[0]' --env.scene.num-envs 4096 --agent.run-name rough
python scripts/train.py Unitree-Go2-AMP-Loco --gpu-ids '[0]' --env.scene.num-envs 4096 --agent.run-name amp
python scripts/train.py Unitree-Go2-AMP-Rough --gpu-ids '[0]' --env.scene.num-envs 4096 --agent.run-name amp_rough
python scripts/train.py Unitree-Go2-PIE --gpu-ids '[0]' --env.scene.num-envs 128 --agent.run-name pie
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

PIE uses two depth frames, ten steps of proprioception and a GRU. It defaults
to 128 training environments to limit visual rollout memory. See
[PIE training, validation and deployment](src/tasks/pie/README.md) for the observation
contract, recurrent ONNX inputs, smoke checks and visual ablations.

## Other scripts

| Script | Description |
| --- | --- |
| `scripts/play_motion_go2.py --motion-file <npz>` | Replay a reference motion clip on Go2 |
| `scripts/validate_pie.py --checkpoint-file <pt>` | CPU recurrent replay and multi-step ONNX parity checks |
| `scripts/evaluate_pie.py --checkpoint-file <pt>` | Fixed-level terrain evaluation with normal, frozen or delayed depth |


scp -r ljl@192.168.1.122:/DATA/rl_ws/go2_mjlab/logs/rsl_rl/go2_amp_rough/2026-10-05_11-09-01_amp_rough logs/rsl_rl/go2_amp_rough

python scripts/play.py Unitree-Go2-AMP-Rough --checkpoint-file logs/rsl_rl/go2_amp_rough/2026-10-05_11-09-01_amp_rough/model_1000.pt
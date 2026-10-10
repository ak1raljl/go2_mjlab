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
| `Unitree-Go2-PIE-Parkour` | PIE with external velocity guidance on randomized obstacles, rough ground, smooth stairs and slopes |

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
| `test/validate_pie.py --checkpoint-file <pt>` | CPU recurrent replay and multi-step ONNX parity checks |
| `test/evaluate_pie.py --checkpoint-file <pt>` | Fixed-level terrain evaluation with normal, frozen or delayed depth |


```bash
python scripts/play.py Unitree-Go2-AMP-Rough --checkpoint-file logs/rsl_rl/go2_amp_rough/2026-10-07_11-40-46_amp_rough/model_24000.pt --keyboard

python scripts/play_pie.py --checkpoint-file logs/rsl_rl/go2_pie/2026-10-07_11-16-08_pie/model_2000.pt  --depth

默认随机选择地形、难度为 **level 0**，每次 reset 采样速度并保持到本回合结束，已关闭 push 等干扰。

可在命令后追加：

- `--terrain gap --terrain-level 5`：查看 level 5 的 gap 地形，难度范围 `0–9`。
- `--speed 0.8`：固定期望速度为 `0.8 m/s`。
- `--keyboard`：使用键盘控制，不能与 `--speed` 同时使用。

`--depth` 显示原始深度和训练预处理后的输入深度，不需要时去掉即可。


python scripts/play_pie.py --checkpoint-file logs/rsl_rl/go2_pie_parkour/2026-10-08_17-05-22_pie/model_5000.pt  --depth  --terrain-level 6

scp -r ljl@192.168.1.122:/DATA/rl_ws/go2_mjlab/logs/rsl_rl/go2_pie_parkour_amp logs/rsl_rl/
```
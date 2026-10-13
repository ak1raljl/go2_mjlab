# Go2 MTE Parkour AMP

新增正式任务：`Unitree-Go2-MTE-Parkour-AMP`。这是对 `Unitree-Go2-PIE-Parkour-AMP` 的融合模块整体替换实验：二维位置编码、全局上下文、单 Query 多头交叉注意力和输出投影一起改变。它不能单独证明多头数量或某一个分支带来收益。

计划依据与执行起点均为 `ff11de34e05d0669ea6009e704fbd1ee6792bd6b`，执行日期 2026-10-13。原始 `Go2_MTE_Parkour_AMP_Codex_Plan.md` 保持原状；本文记录实际实现和验证状态。实际仓库已有 PIE/Parkour/AMP 和 `test/`，优先采用真实源码，未采用 AGENTS.md 中已过时的任务列表说明。

## 实现与边界

- 模型：`src/tasks/pie/rl/mte_model.py:MTEActorModel`。
- 基类：`src/tasks/pie/rl/pie_model.py:PIEActorModel._build_fusion()`。原 Transformer 构造原样移入 hook，原参数名、构造顺序和默认数学路径保持不变。MTE 不创建旧 Transformer。
- 配置：`src/tasks/pie/config/go2/mte_rl_cfg.py:unitree_go2_mte_parkour_amp_runner_cfg()`，深拷贝原 AMP 配置，删除三个无效 `transformer_*` 配置项，采用独立实验目录 `go2_mte_parkour_amp`。
- 注册：`src/tasks/pie/config/go2/__init__.py`。训练和播放环境分别调用原 AMP 环境工厂；未修改相机、地形比例、课程、奖励、终止条件、动作库或 AMP 权重，未恢复 stepping stones。
- 训练：原 `PIEAMPOnPolicyRunner`、`PIEAMPPPO`。保留共享 `forward_with_auxiliary()`、非重入 gradient checkpoint、rollout 起始 hidden state、padding/unpadding 和 done reset。
- 播放：`scripts/play_pie.py --task Unitree-Go2-MTE-Parkour-AMP`，复用路线控制和 AMP 推理降配路径，不需要专家数据和判别器。

### 默认结构

`...` 表示 `[B]` 或循环训练的 `[T,B]`；传感器观测中的深度是展平的，导出接口接收图像形状。

| 模块 | 输入 → 输出 |
| --- | --- |
| 当前本体 | `[...,45]`；仅与估计结果拼接进入 Actor |
| 本体历史 MLP | `[...,450] → 256 → 128 → 64`，沿用历史 normalizer |
| CNN | 两帧 `[2,60,86]` 减 0.5，通道 32/64/64，核 8/4/3，stride 4/2/1，输出 `[64,4,7]` |
| 局部 token L0 | 原 LayerNorm，`[...,28,64]` |
| 位置编码 | 按 CNN flatten 顺序生成 `(x,y)∈[-1,1]²`，`Linear(2,64)` 后与 L0 相加得 L |
| 全局上下文 g | 每个 L 经 `64→32→32`（均 ELU），token 维 `amax` 得 `[...,32]` |
| Query | `[p,g]` 经 `96→64→64`（均 ELU），得到 `[...,1,64]` |
| Attention | `MultiheadAttention(64,8,dropout=0,batch_first=True)`；Q 来自本体+全局，K/V 均来自同一组 L |
| 输出融合 | `[p,g,z]` 为 160 维，`Linear(160,128)→ELU` |
| 原 GRU | 单层，hidden=128，保留 `[1,B,128]` 状态接口 |
| 估计输出 | 速度 3 + 地图 latent 16 + 足端净空 4 + VAE mu 16，共 39 |
| 原 Actor | `45+39=84 → 512 → 256 → 128 → 12` |

网格尺寸从实际 CNN 输出推导，仅初始化一次并注册 nonpersistent buffer，随设备/dtype 迁移。MHA 前合并前导维度，之后恢复。正常训练 `need_weights=False`，不收集注意力权重。没有额外的 Q/K/V 外部投影、视觉 self-attention、逐 token FFN、head gate 或 diversity loss。

这里的“全局”是当前两帧图像的整体上下文，位置编码是二维特征网格坐标；两者都不是世界坐标地图。

深度闭环是“传感器→动作→物理环境→新传感器输入”。GRU 提供内部时间记忆；辅助目标中的下一步本体预测使用真实后继观测，当前高度图重建则监督当前局部地形。没有未来深度预测。保持速度、净空、高度重建、successor、KL 的原权重；跌倒与 timeout 后的 reset 观测均由 `successor_valid` 排除。Actor 使用确定性的 VAE mu，仅辅助 VAE 分支采样。privileged 字段只用于原 Critic 和监督。

## 运行命令

先激活已有环境，以下命令均在仓库根目录执行。无显示环境可使用 `MUJOCO_GL=disable`；记录视频另用 `egl`。GPU 编号应按 `nvidia-smi` 实际空闲情况选择。

```bash
conda activate go2_mjlab
python scripts/list_envs.py --keyword Go2
python scripts/train.py Unitree-Go2-MTE-Parkour-AMP --help

# 有限 smoke：实际训练入口，每次保存保留原 ONNX + 元数据导出。
python scripts/train.py Unitree-Go2-MTE-Parkour-AMP \
  --gpu-ids '[0]' --env.scene.num-envs 64 \
  --agent.max-iterations 20 --agent.save-interval 20 \
  --agent.run-name mte_smoke --enable-nan-guard True

# 头数实验：以同一任务的配置覆盖，不新增任务。
# 把 4 换为 1 或 16；默认 8。其余训练参数照上。
python scripts/train.py Unitree-Go2-MTE-Parkour-AMP \
  --gpu-ids '[0]' --agent.actor.cnn-cfg.attention-heads 4 \
  --env.scene.num-envs 64 --agent.max-iterations 20 \
  --agent.save-interval 20 --agent.run-name mte_heads4_smoke

# 播放/导出：用实际 checkpoint 路径替换下方路径。
python scripts/play_pie.py --task Unitree-Go2-MTE-Parkour-AMP \
  --checkpoint-file outputs/mte_gpu64/mte_training/model_19.pt \
  --num-envs 1 --terrain flat --headless-steps 24 --export
# 交互播放：去掉 --headless-steps 24，可加 --viewer viser。
```

播放自动读取 checkpoint 同级 `params/agent.yaml` 的 head 数。若只复制了权重文件，非默认模型必须提供 `--attention-heads 1/4/16`，且应保留原参数文件。权重矩阵形状本身不能区分 head 数；strict load 不足以检查这种拓扑差异。恢复训练同样必须采用原 head 配置。主动指定的播放 head 数与参数文件冲突时会报错。

原 PIE checkpoint 仅能严格加载到原 PIE；MTE 默认从头训练，不支持把旧 Transformer 权重静默部分加载成 MTE。新旧导出 I/O 契约相同：输入 `proprio, proprio_history, depth_history, memory_h_in`，输出 `actions, memory_h_out`。推理不依赖辅助 target；保持 PIE 契约元数据，并为 MTE 增加 `policy_architecture`、`attention_heads`。

### 验证与独立导出

```bash
python test/validate_mte.py \
  --legacy-checkpoint logs/rsl_rl/go2_pie_parkour_amp/2026-10-11_17-27-48_pie_amp/model_13000.pt \
  --checkpoint-file outputs/mte_gpu64/mte_training/model_19.pt \
  --output outputs/mte_validation_trained.json \
  --export-dir outputs/mte_validation_export

# 未训练模型也可完整执行 CPU 验证；不传 checkpoint 参数即可。
python test/validate_mte.py
# 非默认 head 模型同时提供 --attention-heads 4（或 1/16）。
```

该脚本复用现有 runner ONNX 导出器，并验证真正保存/重载的 `policy_trace.pt`，进行 12 步动作与 hidden 对比，包含显式 reset。容差与原脚本相同：`atol=rtol=1e-4`。独立导出的 ONNX 用于数值验证；带完整环境/传感器元数据的部署产物由 train/save 或 play `--export` 生成。

**已有 JIT 限制**：上游 `export_policy_to_jit()` 使用 `torch.jit.script()`，原 PIE 的 `GaussianDistribution._distribution=None` 导致其 entropy 属性编译失败。此改动没有扩大到重写分布；验证中使用的是明确标记的 `torch.jit.trace`，不能把它称为 script 支持。

### 参数、延迟和真实训练 profile

```bash
# CPU 参数/延迟；两种结构各起一个进程。
python test/profile_mte.py --output-dir outputs/mte_cpu_new

# 执行前 nvidia-smi 检查设备；以下运行 PIE 和 MTE 各 20 次真实更新。
# 每次使用新目录，脚本拒绝覆盖已有训练目录。
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=disable python test/profile_mte.py \
  --training --device cuda:0 --num-envs 64 --iterations 20 \
  --output-dir outputs/mte_gpu64_new
```

profile 调用 `scripts.train.run_train()` 和原 runner.learn()，只包装 act/update 做计时与显存统计；训练算法、env 配置和训练步长未替换。启用原 NaN guard，并保存 config、checkpoint、ONNX。两模型使用相同 seed、地形、24 步 rollout、4 minibatch、5 epoch、float32 和 gradient checkpointing；模型构造后再次固定 torch seed，使初始 episode-length 随机数一致。架构初始化消耗的随机数不同，不声称两种结构共享逐位相同的初始权重。

输出包括 `comparison.json`、`pie.json`/`mte.json`、逐更新 `*_updates.json` 和完整日志。记录提交、工作区差异 hash、源文件 hash、依赖版本、GPU、seed 和测量配置。B=1/64 的推理在 20 次 warmup 后重复 100 次，CUDA 前后同步；训练时间同样同步。首轮包含冷分配，应与后续轮分开解释。

PyTorch allocated/reserved 峰值覆盖 rollout、returns 与真实 PPO/辅助/AMP 更新，排除保存导出阶段。MuJoCo-Warp、仿真/渲染等绕过 PyTorch allocator 的分配不包含在其中。`nvidia-smi` 只提供整卡瞬时快照，包含桌面/其他进程，绝非 allocator 峰值。每轮同时记录 done 数、真实样本、padding slots 和 trajectory 长度，解释模型动作导致的序列差异。

## 实际验证结果

CPU 验证已通过（`outputs/mte_validation_initial.json`）：

- 原始基准提交模型与抽取 hook 后 PIE 的初始化 state_dict 逐位一致；真实旧 AMP `model_13000.pt` strict load 后同输入动作逐位一致。
- 注册和 train/play 环境配置一致，配置工厂不污染旧 task；MTE 无旧 Transformer 参数，1/4/8/16 头可构造，非法头数被拒绝。
- 单步与完整循环序列、非零 rollout 起始 hidden、不同 episode 长度 padding、done reset 和动作 log-prob 回放通过。
- Actor loss 可达 CNN/本体历史/position/global/query/MHA/fusion；共享估计器只调用一次，gradient checkpoint 开关的前向与全梯度一致。
- target 隔离、跌倒/timeout reset successor 标记、有效/无效 target 替换及 padding loss 检查通过。
- MTE 权重保存/重载一致；新初始化模型 12 步 ONNX 最大动作误差 `1.416e-7`、hidden `1.341e-7`；JIT trace 动作和 hidden 误差均为 0。

### 参数口径

| 参数范围 | PIE | MTE | 变化 |
| --- | ---: | ---: | ---: |
| 融合模块 | 99,968 | 50,944 | −49.04% |
| 完整 Actor（含辅助 decoder、logvar、动作 std） | 697,698 | 648,674 | −7.03% |
| 确定性 actions/hidden 依赖参数 | 643,987 | 594,963 | −7.61% |

新融合分项：MHA 16,640；global MLP 3,136；query MLP 10,368；fusion Linear 20,608；position Linear 192。原 depth LayerNorm 是公共部分。完整 Actor 的辅助 decoder + logvar head 共 53,699，动作 std 为 12；最后一行排除它们。原 eager 估计器仍会计算随后丢弃的 logvar；导出 wrapper 的 deepcopy 仍携带完整 Actor，不能把路径参数数当作 wrapper 参数或文件体积。

### 64 环境 GPU 实测

设备：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB；Python 3.11.17、PyTorch 2.14.1 / CUDA 13.0、mjlab 1.2.0、MuJoCo / MuJoCo-Warp 3.5.0、Warp 1.12.0、RSL-RL 5.0.1、ONNX Runtime 1.27.0。seed=42，episode 长度保留原 40 秒，其他配置见上文。两种模型各以独立进程完成 20 次更新，每轮 1,536 个真实 transition，启用 NaN guard；所有 PPO、五项辅助损失与 AMP/gradient-penalty 指标有限。

| 测量项 | PIE | MTE |
| --- | ---: | ---: |
| GPU 推理 B=1（ms/次） | 0.828 | 0.661 |
| GPU 推理 B=64（ms/次） | 1.158 | 0.712 |
| 20 轮最高 allocated（MiB） | 760.67 | 774.44 |
| 20 轮最高 reserved（MiB） | 794.00 | 814.00 |
| 更新耗时均值，排除首轮（秒） | 0.4024 | 0.4081 |
| 更新耗时中位数，排除首轮（秒） | 0.3733 | 0.3532 |
| rollout+returns+update 均值，排除首轮（秒） | 1.1963 | 1.2166 |
| 每轮 padding slots 范围 | 0–192 | 24–216 |
| 20 轮累计 done transition 数 | 50 | 93 |
| 训练保存 ONNX 文件（bytes，含元数据） | 2,630,920 | 2,407,762 |

**本次没有观察到峰值训练显存降低**，也没有稳定的更新吞吐提升证据。两种模型最高 allocated 分别发生在第 9/17 轮，恰与各自最大 padding（192/216）同轮；动作、终止与轨迹差异是混杂因素，不能把本次差异完全归因于网络。每轮更新后的 allocated 分别稳定在约 323.4–323.5 / 322.6–322.7 MiB，未见本次 20 轮内的持续增长。桌面 GPU 上一次有限对比不构成普遍性能结论；未扩大到 256 环境。

整卡 `nvidia-smi` 运行前约 960/961 MiB，运行后快照约 2237/2184 MiB；这包含其他分配与进程，不能当作上述 allocator 峰值或进程峰值。

CPU 双线程推理参考值（B=1/B=64，ms/次）：PIE 1.032/9.375，MTE 0.628/9.188；原始结果为 `outputs/mte_cpu/comparison.json`。CPU 与 GPU 测量只比较同设备数据。

完整 GPU 数据：`outputs/mte_gpu64/comparison.json` 与 `*_updates.json`。本次已保存：

- `outputs/mte_gpu64/pie_training/model_19.pt`
- `outputs/mte_gpu64/mte_training/model_19.pt`
- 两目录各自的 `policy.onnx`、`params/agent.yaml`、`params/env.yaml`、TensorBoard 日志。

训练后 `model_19.pt` 再次通过完整 CPU 验证（`outputs/mte_validation_trained.json`）：12 步 ONNX 最大动作误差 `8.345e-7`、hidden `2.533e-7`；JIT trace 两项误差仍为 0。`outputs/mte_validation_export/` 含实际导出的 ONNX 与 trace 文件。

实际执行 `play_pie.py` 严格加载该 checkpoint、自动读取 8 头配置，在 flat 地形完成 24 步 headless 播放，并重新导出带元数据的 ONNX。结果在 `outputs/mte_play_stats.json`。此 0.48 秒播放未完成一个 episode，不能作为 route 成功率。

### 实施验收状态

- [x] 新任务 registry / 训练 CLI 可加载，且只新增一个正式 task。
- [x] 默认 8 头单 Query 输出 128 维接原 GRU，无旧 Transformer 参数。
- [x] 环境、CNN、本体编码器、GRU、Actor、辅助目标及 AMP 均复用基线。
- [x] 原模型初始化与真实旧权重输出回归，配置隔离、梯度、循环状态和监督隔离通过。
- [x] MTE 权重往返、训练后 ONNX / JIT trace 多步一致性及实际播放通过。
- [x] 64 环境、每模型 20 次真实更新及资源测量完成，文档记录实际结果。
- [ ] 上游 `torch.jit.script` 路径未修复，原因见已有 JIT 限制；ONNX 和 JIT trace 已验证。

所有 `outputs/` 文件均为本地、gitignored 运行产物。无需要用户补充的设备或数据阻塞；剩余长期行为评测属于下面的后续范围。

## 后续评测边界

本次短训练只验证管线，不证明收敛、落脚精度或后脚改善。`test/evaluate_pie.py` 的旧距离/存活代理指标不能用于这些结论。后续需要实现真实 route 完成率、前脚成功条件下后脚失败率、落脚边缘裕量、落地后一秒失稳率，并在固定难度/速度和多个随机种子下比较。

未执行长期训练、256/2048/4096 环境扩展、分辨率/帧数/CNN 修改、奖励或 AMP 调整、前后脚专用记忆、边缘辅助损失、长期历史缓存或实机部署。未推送、合并或发布。

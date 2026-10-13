# Go2 MTE Parkour AMP：Codex 执行计划

## 0. 执行指令与交付范围

请在 `go2_mjlab` 仓库实现本计划，新增且仅新增以下正式 task：

```text
Unitree-Go2-MTE-Parkour-AMP
```

目标：在现有 `Unitree-Go2-PIE-Parkour-AMP` 基础上，把视觉—本体融合模块替换为轻量全局—局部多头交叉注意力。保留训练环境、CNN、本体历史编码器、GRU、状态估计头、辅助训练目标、Actor、Critic 和 AMP 机制，建立可公平比较的结构基线。

本文件是实施计划，所有验收项初始均未完成。执行者应直接完成代码、注册、必要测试、可行的短训练与文档；不要只回复方案。不要自动开始完整训练、推送远程、合并或部署实机。

仓库：<https://github.com/ak1raljl/go2_mjlab.git>

计划依据：`ff11de34e05d0669ea6009e704fbd1ee6792bd6b`，编写日期：2026-10-13。执行时以实际工作区为准，先核对差异，不要为了匹配该提交回退用户代码。

## 1. 先检查实际代码

1. 阅读适用的 `AGENTS.md`，检查 `git status`、当前分支和提交，保留用户未提交修改。
2. 优先使用已有仓库；不要在不清楚当前目录的情况下重复克隆。若无本地仓库，再获取上述仓库。
3. 阅读以下关键路径，确认真实接口；若目录已变动，使用 `rg` 查找对应实现：

```text
src/tasks/__init__.py
src/tasks/pie/config/go2/__init__.py
src/tasks/pie/config/go2/rl_cfg.py
src/tasks/pie/config/go2/amp_rl_cfg.py
src/tasks/pie/config/go2/amp_env_cfg.py
src/tasks/pie/config/go2/parkour_env_cfg.py
src/tasks/pie/rl/pie_model.py
src/tasks/pie/rl/ppo.py
src/tasks/pie/rl/amp_ppo.py
src/tasks/pie/rl/amp_runner.py
src/tasks/pie/mdp/observations.py
scripts/train.py
scripts/play.py
scripts/play_pie.py
test/validate_pie.py
```

已核对的基线事实：

- 当前本体观测 45 维，历史 10 步，共 450 维。历史展平可能按 observation term 组织，不要直接假设可以 reshape 为 `[B,10,45]`。
- 深度输入是两帧组成通道的 `[B,2,60,86]`，不是 `60×80`。
- CNN 通道 `(32,64,64)`，卷积核 `(8,4,3)`，步幅 `(4,2,1)`，无 padding；输出 `[B,64,4,7]`，即 28 个 64 维 token。
- 原融合模块是两层 `TransformerEncoderLayer`，维度 64、单头、FFN 隐层 256。
- 原融合输出为 128 维，接一层 hidden size 128 的 GRU。
- 当前模型有共享 Actor/辅助损失前向和梯度检查点，必须保留。
- AMP task 从基础 runner config 构造，使用 `PIEAMPOnPolicyRunner` 和现有 AMP PPO。

如果 HEAD 已有同名 MTE task，先检查并补齐它，不重复注册。如果网络维度已变化，记录差异并保持新旧实验输入一致；仅在无法兼容的核心设计冲突时询问用户。

## 2. 第一版固定设计

### 2.1 输入和输出

输入、动作含义及动作分布与原 task 一致：

| 张量 | 默认形状 | 说明 |
| --- | --- | --- |
| 当前本体 `proprio` | `[B,45]` | 原 Actor 输入 |
| 本体历史 `history` | `[B,450]` | 原历史 MLP 输入 |
| 深度历史 `depth` | `[B,2,60,86]` 或原观测中的展平形式 | 沿用现有预处理 |
| GRU 隐状态 | `[1,B,128]` | 沿用原循环策略接口 |
| 动作 | `[B,12]` | 原动作分布、缩放和执行方式 |

网络还须支持循环训练中的 `[T,B,...]` 前导维度。视觉干路可暂时合并前导维度，送入 GRU 前必须恢复，不可混淆时间与 batch。

### 2.2 局部视觉 token 和位置编码

沿用原 `_encode_depth()` 的减去 0.5、CNN 和 token LayerNorm，得到 `L0`，形状 `[...,28,64]`。

在 CNN 输出网格建立二维归一化坐标：横纵坐标范围均为 `[-1,1]`，展平顺序严格匹配 CNN feature 的 flatten 顺序。用 `Linear(2,64)` 得到位置编码：

```python
L = L0 + position_proj(grid_xy)  # [..., N, 64]
```

从 CNN 输出尺寸推导网格，默认 `4×7`，不要把 28 或原始图像坐标硬编码到 forward。坐标网格可用非持久 buffer 注册，随模型迁移设备和 dtype。不要在每次 forward 用 Python 重建网格。

此坐标编码是特征图位置，不是三维地图坐标。

### 2.3 本体编码与全局上下文

本体历史编码器保持 `450→256→128→64`，沿用原观测归一化和激活：

```python
p = history_encoder(history_normalizer(history))  # [..., 64]
g_tokens = global_mlp(L)                          # [..., N, 32]
g = g_tokens.amax(dim=-2)                         # [..., 32]
```

`global_mlp` 固定为 `Linear(64,32) → ELU → Linear(32,32) → ELU`。

这里的全局特征是当前两帧输入的整体上下文；不代表地形类别标签，也不是世界坐标系全局地图。第一版不增加长深度历史、第二个 CNN 或显式建图模块。

### 2.4 单 Query 多头交叉注意力

```python
q = query_mlp(torch.cat([p, g], dim=-1)).unsqueeze(-2)
z, _ = attention(query=q, key=L, value=L, need_weights=False)
z = z.squeeze(-2)
```

- `query_mlp`：`Linear(96,64) → ELU → Linear(64,64) → ELU`。
- `attention`：`nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.0, batch_first=True)`，默认 bias=True。
- 原生 MHA 内部完成 Q/K/V 投影，不要再无意增加重复的外部投影。
- MHA 输入先合并前导维度，输出后恢复。
- 默认 **8 头**；支持通过配置改为 1、4、16 头，不额外注册 task。校验 `token_dim % attention_heads == 0`。
- 不引入视觉 self-attention，不堆第二层，不加入原来的逐 token `64→256→64` FFN。
- 第一版不加入 head gate、强制头分工或 attention diversity loss，避免同时引入多个变量。
- 正常训练禁止返回/保存注意力权重；离线诊断可以显式开启，但默认关闭。

Q 由本体和全局特征共同生成，K/V 均来自相同局部 token 集合；不能分别使用无对应关系的全局 K 和局部 V。

### 2.5 融合、GRU 和动作输出

```python
memory_input = fusion_proj(torch.cat([p, g, z], dim=-1))
# fusion_proj: Linear(160,128) → ELU
```

然后完整复用原策略的 GRU、估计头和 Actor 路径：

```text
128 维融合结果 → GRU(128)
→ 速度 3 + 地图 latent 16 + 足端净空 4 + VAE mu 16
→ 39 维估计结果，与当前本体 45 维拼接
→ Actor MLP 84→512→256→128→12
```

动作使用确定性的 VAE mu 构造条件分布；辅助 VAE 分支才采样，不得改变 PPO 采集和更新时条件动作分布的一致性。

## 3. 深度闭环与辅助目标：完整继承

本次不预测下一帧深度。深度是传感器输入；动作执行后环境提供新深度与本体观测，形成真实控制闭环。GRU 隐状态构成策略内部的记忆回路。

保持现有辅助目标及权重：速度、足端净空、当前高度图重建、下一步本体预测、KL。下一步本体 target 仍在 `process_env_step` 取真实后继观测；所有 episode 边界（含 timeout）后 reset 观测不得当作前一状态的合法后继，保留 `successor_valid`。

监督真值只进入辅助 loss 或原 Critic，不能进入 Actor 特征计算。不要增加前脚接触记忆、后脚专用分支、边缘监督、未来图像解码或新奖励。

## 4. 文件组织与兼容实现

建议新增：

```text
src/tasks/pie/rl/mte_model.py
src/tasks/pie/config/go2/mte_rl_cfg.py
test/validate_mte.py
test/profile_mte.py
docs/mte_parkour_amp.md
```

具体布局可跟随当前仓库惯例，但不要复制整个环境和 PPO 实现。

建议 `MTEActorModel` 继承 `PIEActorModel`，复用估计、循环状态、辅助训练、分布和导出接口。为避免生成无用的旧 Transformer，可在基类增加一个最小的融合模块构建 hook：基类默认实现保持旧模块及参数名；子类 override 构建 MTE 模块并 override `_trunk()`。hook 在依赖的维度及编码器初始化后调用，并且子类不能访问尚未初始化的成员。

硬性要求：

1. MTE 模型不能保留未使用的旧 Transformer 参数，也不能把两套干路都交给优化器。
2. 原 PIE 默认计算路径、参数名称、shape、初始化顺序尽可能保持原状；旧 task 的 checkpoint 必须继续严格加载。
3. 新模型保持 `forward_with_auxiliary()` 的共享一次估计器前向，不可重复计算 CNN/GRU。
4. checkpoint 仍包装新 `_trunk()`，保留 `use_reentrant=False`；不要 detach 编码特征来伪造显存节省。
5. 保留 recurrent masks、padding/unpadding、rollout 起始 hidden state 和 done reset 逻辑。
6. 新旧模型融合权重不兼容；默认新 task 从头训练。不要静默 `strict=False` 将旧 checkpoint 当成完整 MTE checkpoint。
7. 导出仍使用 `proprio, proprio_history, depth_history, memory_h_in` 四个输入，返回 `actions, memory_h_out`；推理不得依赖辅助 target 字段。

## 5. 配置和任务注册

在 `mte_rl_cfg.py` 新增配置工厂，例如 `unitree_go2_mte_parkour_amp_runner_cfg()`：

1. 调用现有 `unitree_go2_pie_parkour_amp_runner_cfg()` 后 deepcopy，避免共享嵌套字典污染旧 task。
2. 修改 `cfg.actor.class_name` 为 `src.tasks.pie.rl.mte_model:MTEActorModel`。
3. deepcopy `cfg.actor.cnn_cfg`，设置 MTE 字段，例如：

```python
token_dim = 64
attention_heads = 8
global_dim = 32
memory_hidden_dim = 128
gradient_checkpointing = True
```

4. 继承 CNN、depth_shape、history、Actor、Critic、PPO、AMP、训练步数等原配置；清除仅属于旧 Transformer 的失效配置键或在配置文档中明确标为不适用，不能让用户误认为改它们有效。
5. `cfg.experiment_name = "go2_mte_parkour_amp"`，确保日志和 checkpoint 不混入旧实验目录。

在已有 `src/tasks/pie/config/go2/__init__.py` 注册新 task；采用新的配置实例，不复用可变的注册对象：

```python
register_mjlab_task(
    task_id="Unitree-Go2-MTE-Parkour-AMP",
    env_cfg=unitree_go2_pie_parkour_amp_env_cfg(),
    play_env_cfg=unitree_go2_pie_parkour_amp_env_cfg(play=True),
    rl_cfg=unitree_go2_mte_parkour_amp_runner_cfg(),
    runner_cls=PIEAMPOnPolicyRunner,
)
```

验证 `import src.tasks` 后 registry 能查询该 task，且旧 task 没有消失或被覆盖。若现有包自动发现机制要求补充导入，按仓库惯例修改。

为 `scripts/play_pie.py` 的 task 选择增加新 ID，并检查其下游是否硬编码模型/runner；通用 play 能使用则优先复用。历史文件名可保留。所有训练环境参数，包括相机、地形比例、奖励、课程、动作库、AMP 权重和终止条件，第一版均不变；当前排除的 stepping stones 不恢复。

## 6. 结构成本与显存测量

以下是本计划默认网络的理论核对值，不是实测性能：

| 融合模块 | 参数量 |
| --- | ---: |
| 原两层 Transformer | 99,968 |
| 新 MHA | 16,640 |
| 全局 MLP | 3,136 |
| Query MLP | 10,368 |
| 输出融合 Linear | 20,608 |
| 位置 Linear | 192 |
| 新融合合计 | 50,944 |

旧的深度 LayerNorm 属于共同部分，不重复计入。若实现另加 LayerNorm 等模块，应如实更新计数。默认结构融合参数约减少 49%；整个策略推理路径参数仅约减少 7.6%，不能宣传为整网或显存减少 49%。辅助解码器可能仍存在于导出 wrapper 的 deepcopy 模型中，要区分路径实际使用参数、完整模型参数和导出文件体积。

`test/profile_mte.py` 至少提供以下报告：

- 原 task 与新 task 的完整 Actor、融合模块参数数及推理路径参数数，明确辅助解码器是否计入。
- 相同 batch 下的策略推理延迟，使用 warmup 和 CUDA synchronize；分别给 B=1 和一个批量场景。
- 相同 env 数、rollout 长度、minibatch、精度及 gradient checkpointing 下的真实训练峰值 allocated/reserved 显存和更新耗时；不能只比较随机张量前向。
- 记录 GPU、软件版本、提交、seed 和测量配置。CUDA 无法统计的仿真/渲染分配与设备整体显存要单独说明，不要与 PyTorch allocator 数据混用。

不同 task 最好各用新进程测量，避免缓存污染。使用同一 terrain/seed 和初始配置；实际 rollout 终止数量、padding 长度可能不同，要同时报告有效样本/填充量以解释差异。

测量前检查 GPU 使用情况，不占用繁忙设备进行长任务。先用 64 个环境；资源充足时可追加一次 256 环境比较，不自动升到 2048/4096。一次有限对比足够，除非有具体故障，不反复扩大测试。

若资源不足，完成 CPU 结构测试并明确 GPU 项未运行。不得凭注意力矩阵尺寸推断实际显存下降。

## 7. 必要验证

### 7.1 模型和训练语义

- 验证单步 `[B,...]` 和循环 `[T,B,...]` 路径输出、动作分布、辅助 loss 与 hidden state 的 shape 和有限性。
- 验证 actor loss 能给 CNN、global/query/fusion MLP 和 MHA 传递梯度，防止错误 detach 或旁路。
- 使用固定参数、同一输入与相同初始 hidden，比较逐步 recurrent 推理和完整无 padding 序列计算；再覆盖不同长度 padding 和 episode reset，防止跨 episode 状态泄漏。
- 比较 checkpoint 开/关的前向结果及梯度，随机采样的辅助分支要固定随机状态或隔离确定性部分。
- 在输入、hidden 和确定性动作路径固定时，只改变 privileged target 字段，动作不变；辅助 loss 可以变化。
- 构造跨 reset 的 successor 样本，验证 masked loss 不把 reset 帧当后继。

### 7.2 兼容和导出

- 验证原 task 配置没有被新配置工厂污染，原 actor 不多出 MTE 权重。
- 有旧 checkpoint 时用原 task strict load，并做相同输入的输出回归；没有时说明缺失，不把合成 state_dict 测试写成真实旧权重验证。
- MTE checkpoint 保存/加载后，确定性动作和隐状态一致。
- 复用现有导出工具，验证 PyTorch 与 ONNX/JIT 的多步动作/hidden 一致性，包含至少一次显式 reset；沿用已有容差并报告最大误差。
- 若 MHA 导出遇到后端 fastpath 不支持，优先采用与现有导出器兼容的计算路径；不得仅为了导出重写训练数学或静默绕过注意力。缺少 ONNX 运行依赖时如实报告。

### 7.3 短训练

先确认真实 CLI help，以下命令为预期用法，参数若有差异应按代码修正并写入文档：

```bash
python scripts/train.py Unitree-Go2-MTE-Parkour-AMP --help

python scripts/train.py Unitree-Go2-MTE-Parkour-AMP \
  --gpu-ids '[0]' \
  --env.scene.num-envs 64 \
  --agent.max-iterations 20 \
  --agent.run-name mte_smoke
```

使用确认空闲的设备，GPU 0 只是示例。检查短训练完整覆盖 rollout、循环 PPO update、辅助损失与 AMP update，无 NaN、维度错误或异常显存增长，并保存可用于加载/播放的 checkpoint。必要时设置现有 save 配置，不增加新训练机制。

短训练只证明管线可运行，不用于宣称落脚改善或收敛。资源或数据缺失时，保留明确的复现命令和未执行原因，不伪造日志。

## 8. 文档与后续实验边界

`docs/mte_parkour_amp.md` 至少包括：

1. Task ID、实际代码入口、各模块张量维度、Q/K/V 来源。
2. 全局上下文和二维位置编码的含义；不声称实现世界地图。
3. 深度控制闭环、GRU 记忆、下一步本体预测及当前高度重建的区别。
4. 真实可执行的 train/play/export/validate/profile 命令，heads 如何调整为 1/4/16。
5. 参数与实测资源对比；未测项目标注“未测”。
6. 与原网络 checkpoint 的兼容范围及从头训练要求。

本次不要求重写完整地形评测系统。当前 `test/evaluate_pie.py` 固定基础任务且使用距离/存活代理指标，不能拿其结果直接宣称新 Parkour task 后脚性能提高。在文档登记后续评测需要：真实 route 完成率、前脚成功条件下后脚失败率、落脚边缘裕量、落地后一秒失稳率，以及固定难度/速度和至少多个随机种子的比较。未实现的指标不得伪造。

本次不自动执行：完整长期训练、相机分辨率/帧数调整、CNN 缩窄、奖励和 AMP 调整、前后脚记忆、边缘辅助损失、长期历史缓存、实机部署。将这些与本次结构实验隔离。

## 9. 完成标准与执行反馈

- [ ] 新 task 可通过 registry 和训练 CLI 加载，名称精确为 `Unitree-Go2-MTE-Parkour-AMP`。
- [ ] 默认 8 头单 Query 交叉注意力实现，输出 128 维接原 GRU。
- [ ] 新模型没有未使用的旧 Transformer 参数，旧 task 保持原行为。
- [ ] 环境、CNN、GRU、Actor、辅助目标、AMP 与基线一致。
- [ ] 模型梯度、循环状态和监督隔离验证通过。
- [ ] Checkpoint 与导出验证完成，或准确列出不可执行的依赖项。
- [ ] 短训练和资源测量完成，或明确记录设备/数据阻塞。
- [ ] 文档提供实际命令、成本口径及结果；不把短训练当运动性能结论。

最后向用户报告：改动文件、结构差异、已执行验证及结果、未完成项和原因、训练/播放命令。执行计划与代码完成状态必须分开。未经要求不要进行远程 push 或发布。

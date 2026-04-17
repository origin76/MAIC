# 目前完整进度 PPT 大纲

日期：2026-04-16

适用场景：
- 本科毕业设计中期汇报
- 论文实验进展答辩
- 导师组会汇报

建议时长：
- 主体 14 页到 16 页，控制在 12 到 18 分钟
- 附录 4 页到 6 页，放配置表、seed 结果和补充诊断图

建议主线：
- 先证明“为什么要研究通信”
- 再证明“通信不是天然有效，所以必须先做原型验证”
- 接着说明“为什么最后选 MAPPO 做主线”
- 然后讲“如何先把 backbone 做强，再往上叠轻量通信”
- 最后按 v1 到 v5 讲结构演化、实验结论和当前未解问题

## 一、建议页序

| 页码 | 标题 | 这一页要讲什么 | 建议图 |
| --- | --- | --- | --- |
| 1 | 题目与研究定位 | 课题是“基于信息交互的多智能体强化学习算法研究”，核心关键词是 CTDE、信息编码、交互对象选择、信息融合策略 | 无，题目页即可 |
| 2 | 研究目标与总体问题 | 强调三件事：为什么通信值得研究、为什么现有通信不稳定、为什么要做结构化改进 | 可用 [`ppt_algorithm_evolution_timeline.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_algorithm_evolution_timeline.svg) 做缩略图 |
| 3 | 为什么先在 `join1` 做轻量通信验证 | 说明 `join1` 是低成本原型环境，用于先验证“通信是否可能有效”，避免一上来就在 SMAC 上混淆 backbone 问题与通信问题 | 可插 [`join1_test_winrate_comparison.png`](/Users/zerick/code/MAIC/paper/figures/generated/join1_test_winrate_comparison.png) |
| 4 | `join1` 的核心结论 | 重点讲两句话：受约束的稀疏通信可以明显提升；结构不当的通信会直接崩掉 | 继续用 [`join1_test_winrate_comparison.png`](/Users/zerick/code/MAIC/paper/figures/generated/join1_test_winrate_comparison.png) |
| 5 | 为什么最终选 MAPPO 做主线 | MAPPO 更适合做“集中训练、分散执行”下的 actor 侧通信改造；集中 critic 吸收全局信息，actor 保持分布式执行；PPO 稳定器更方便逐步叠结构 | 可用 [`ppt_mappo_backbone_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_mappo_backbone_structure.svg) 局部截图 |
| 6 | MAPPO Backbone 的构建过程 | 讲从普通 vanilla 到 `officialish`，再到 warm-start control 的过程，说明 backbone 不强时不适合研究通信 | 用 [`baseline_officialish_vs_warmstart.png`](/Users/zerick/code/MAIC/paper/figures/generated/baseline_officialish_vs_warmstart.png) |
| 7 | 当前最强 backbone 的完整网络结构 | 这一页是方法页核心，要把 shared actor、centralized critic、PPO 稳定器全部讲清 | 用 [`ppt_mappo_backbone_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_mappo_backbone_structure.svg) |
| 8 | 版本演化总览 | 用一页先总览从 `join1 -> backbone -> v1 -> v5` 的结构递进逻辑，让后面每一页都有上下文 | 用 [`ppt_algorithm_evolution_timeline.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_algorithm_evolution_timeline.svg) |
| 9 | 第一版：最小侵入残差通信 | 讲 `v1` 的目标不是赢，而是验证“零初始化、低带宽、低增益、残差式 adapter 能否安全接入强 backbone” | 用 [`ppt_v1_v3_residual_family.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v1_v3_residual_family.svg) |
| 10 | 第二版：路由锐化通信 | 讲 `v2` 保持 `v1` 主体不变，只改 routing，试图解决“平均听谁都一点”的问题 | 继续用 [`ppt_v1_v3_residual_family.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v1_v3_residual_family.svg) |
| 11 | 第三版：动作意图共享通信 | 讲 `v3` 的关键创新不是更强的 attention，而是把消息从抽象 hidden state 改成明确的 action intention | 继续用 [`ppt_v1_v3_residual_family.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v1_v3_residual_family.svg) |
| 12 | 第四版：攻击子空间定向融合 | 讲 `v4` 为何是一个重要转折点：不再改整个 hidden，而只改攻击 logits；消息有了明确物理语义和明确作用位置 | 用 [`ppt_v4_targeted_fusion_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v4_targeted_fusion_structure.svg) |
| 13 | 第五版：攻移双流通信 | 讲 `v5` 的核心假设：集火和走位不是同一种通信问题，因此要拆成两条流单独建模 | 用 [`ppt_v5_dualstream_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v5_dualstream_structure.svg) |
| 14 | v1 到 v5 的总体结果对比 | 给出每类算法的代表结果，回答“哪些设计是有效的，哪些只是好看但不稳” | 用 [`v1_v5_variant_summary.png`](/Users/zerick/code/MAIC/paper/figures/generated/v1_v5_variant_summary.png) 或 [`v1_v5_family_best_summary.png`](/Users/zerick/code/MAIC/paper/figures/generated/v1_v5_family_best_summary.png) |
| 15 | 机制诊断：攻击流 vs 移动流 | 讲当前最强的机制性发现：攻击流已经能学到明确语义，移动流仍然容易退化为对称平均 | 用 [`communication_diagnostics_attack.png`](/Users/zerick/code/MAIC/paper/figures/generated/communication_diagnostics_attack.png) 和 [`communication_diagnostics_move.png`](/Users/zerick/code/MAIC/paper/figures/generated/communication_diagnostics_move.png) |
| 16 | 当前结论与后续工作 | 收束全文：backbone 必须先做强；轻量通信是可行的；语义和融合位置比单纯加容量更重要；下一步继续解 move stream | 可再放一次 [`ppt_algorithm_evolution_timeline.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_algorithm_evolution_timeline.svg) 或者只用总结表 |

## 二、每页建议讲法

### 第 1 页：题目与研究定位

建议口径：
- 本课题关注多智能体协作任务中的信息交互问题。
- 目标不是简单堆一个更复杂的通信网络，而是回答三个更基础的问题：传什么、听谁、怎么融。
- 整个工作遵循 CTDE 范式，在训练时利用全局信息，在执行时保持分散决策。

### 第 2 页：研究目标与总体问题

建议口径：
- 现有通信方法往往默认“只要能通信就会更强”，但实际并非如此。
- 通信模块至少会遇到三类问题：消息没有语义、路由过于平均、融合位置不合理。
- 因此本文采用逐步演化的方法，不一次性设计大模型，而是每次只改一个结构因素。

### 第 3 到 4 页：`join1` 轻量通信认证

建议口径：
- `join1` 的作用不是拿最终成绩，而是做通信原型验证。
- 这里最重要的结论是：通信不是天然有效的增强器，结构不当时反而会破坏训练。
- 这也是后续在 SMAC 上坚持“低带宽、低增益、残差式”的原因。

你可以明确讲出两句结论：
- 受约束的稀疏通信在简单协作环境中可以带来明显收益。
- 缺乏结构约束的通信会出现几乎完全失效的情况。

### 第 5 页：为什么选 MAPPO

建议讲四点：
- 第一，MAPPO 天然符合“集中训练、分散执行”的 thesis 叙事。
- 第二，actor 是显式策略网络，便于直接研究信息如何影响动作决策。
- 第三，centralized critic 可以提供更稳定的训练信号，把全局信息留在训练端。
- 第四，PPO 的裁剪、KL 监控、value clipping 等稳定器，适合做逐版本结构消融。

可以顺手补一句：
- 这也是为什么最终没有把主要精力放在 value decomposition 主线上，而是回到 MAPPO backbone 上继续做通信设计。

### 第 6 到 7 页：Backbone 构建与完整结构

这一段建议讲成“先把地基做对，再讨论通信”。

Backbone 结果建议直接说：
- `officialish` 直接训练三 seed 平均 `peak = 0.2917`，`final = 0.1250`，`last5 = 0.2042`。
- warm-start control 三 seed 平均 `peak = 0.8542`，`final = 0.6250`，`last5 = 0.6417`。

因此这两页的核心结论是：
- 当前代码框架中的 vanilla MAPPO 不是不能学，而是必须把实现细节和训练稳定器调对。
- 只有 backbone 足够强，通信模块的收益和失败才具有分析价值。

第 7 页建议专门强调 backbone 的三个层次：
- Actor：共享参数的 `Linear -> GRUCell -> MLP policy head`
- Critic：agent-wise centralized critic，输入为 `state + local obs + last action + agent id`
- PPO 稳定器：active masks、ValueNorm、Huber、value clipping、线性 LR 衰减、KL 早停

### 第 8 页：版本演化总览

这一页不要讲具体数值，主要讲演化逻辑：
- `v1` 验证安全接入
- `v2` 尝试让 routing 更尖锐
- `v3` 改变消息载体，让消息更有物理意义
- `v4` 改变融合位置，只修改攻击子空间
- `v5` 拆解攻击协同和移动协同

这一页的任务是给后面 `v1-v5` 五页建立全局框架。

### 第 9 页：第一版，最小侵入残差通信

结构变化要点：
- backbone 完全保留
- 从 `h_i` 派生 query、key、value
- 稀疏 top-k 注意力
- `LayerNorm(message)` 后走小门控和零初始化 residual projection
- 最后以残差形式回加到 actor hidden 上

建议强调：
- `v1` 的关键词不是“强”，而是“稳”
- 它回答的是：通信能不能以极小代价接入一个已经很强的 local policy，而不把它毁掉

### 第 10 页：第二版，路由锐化

结构变化要点：
- 主体结构与 `v1` 一致
- 通过极小的 attention entropy penalty 或更快的 comm 学习速度，逼 attention 停止平均化

建议强调：
- `v2` 不是新网络，而是“只改路由，不改融合”
- 它回答的是：如果消息还是同样的消息，只让路由更尖锐，会不会更有用

### 第 11 页：第三版，动作意图共享

结构变化要点：
- 最大改动是 carrier，不再传 hidden feature，而是传 action intention
- 也就是把“我要打谁 / 我要怎么动”的局部意图显式编码成消息

建议强调：
- 这是一个很适合答辩讲的转折点，因为它把“黑箱消息”改成了“可解释消息”
- `v3` 证明，很多时候不是注意力公式不行，而是传的内容本身没有物理意义

### 第 12 页：第四版，攻击子空间定向融合

这一页建议重点讲公式级思想：
- 先由 backbone 得到本地 logits
- 将 logits 切成 `move/base` 和 `attack`
- 通信只生成 `delta_attack`
- 最终只修正攻击子空间：
  `final_attack = local_attack + scale * gate * delta_attack`

这一页的核心价值在于：
- 通信的作用位置第一次变得非常清楚
- 这使得结果更容易解释，也更适合写论文

建议点出：
- `v4` 是从“加一个通信模块”转向“设计一个有物理含义的信息融合机制”的关键一步

### 第 13 页：第五版，攻移双流通信

这一页建议围绕“为什么要双流”来讲：
- 集火是离散、尖锐、强同步的协作问题
- 走位是局部拓扑、邻居避碰、风险感知的问题
- 这两者的最优消息和最优路由不应该相同

结构变化要点：
- attack stream：延续 `v4` 的 relation-aware targeted fusion
- move stream：单独的 query/key/value、单独的 gate、单独的 delta_move
- 最终分别修正 move logits 和 attack logits，再拼回总 logits

### 第 14 页：v1 到 v5 的结果对比

建议用“阶段性贡献”来讲，而不是只按胜率排座次。

你可以按下面口径总结：
- `v1`：证明轻量通信能安全接入
- `v2`：证明平均路由确实是问题
- `v3`：证明消息语义比单纯 feature 更关键
- `v4`：证明定向、子空间级融合最有解释性，也最容易做成稳定结果
- `v5`：证明攻击协同和移动协同需要拆开看

如果你想在这一页放数值，建议用论文当前表格作为主版本：

| 类别 | 代表实现 | peak | final | last5 |
| --- | --- | ---: | ---: | ---: |
| Backbone | warm-start control | 0.854 | 0.625 | 0.642 |
| 最小侵入通信 | 端到端残差接入 | 0.813 | 0.646 | 0.563 |
| 锐化路由通信 | 软锐化路由 | 0.854 | 0.667 | 0.583 |
| 动作意图共享 | 端到端动作意图共享 | 0.938 | 0.563 | 0.600 |
| 攻击子空间融合 | 基础目标融合 | 0.813 | 0.750 | 0.613 |
| 攻移双流通信 | Top-1 move + softplus | 0.771 | 0.688 | 0.588 |

如果想补最新进展，可以在角落补一行：
- 当前 `v5` 新 sweep 中，`top2move_softplus` 作为多 seed 延续线，均值 `peak = 0.8125`，`final = 0.6042`，`last5 = 0.5958`。

### 第 15 页：机制诊断，攻击流与移动流

这一页建议讲出目前最重要的机制性发现：
- 攻击流已经能学出低熵、尖锐的路由，说明“谁来提供集火信息”是可学的
- 移动流更难，当前 top-2 move 往往退化成近似 `ln(2)` 的对称平均

你可以直接讲当前结论：
- 现在的瓶颈不是“通信有没有容量”
- 也不是“攻击流有没有语义”
- 而是 move stream 如何从两个邻居之间做出非对称选择

这页是答辩里最能体现“你真的理解机制”的地方。

### 第 16 页：结论与下一步

建议最后收成四条：
- 通信机制必须建立在强 backbone 之上，否则结论不可解释
- 轻量、低增益、残差式通信是可靠可行的
- 明确语义的消息比抽象 hidden feature 更容易学出作用
- 子空间级融合和双流解耦提供了更强的结构可解释性

下一步建议讲两件事：
- 继续围绕 `v5` 的 move stream 做 top-2 对称性打破
- 在保持 backbone 不变的前提下，做更克制的稀疏路由改进和更多 seed 验证

## 三、可直接复用的结构图清单

本次新增的矢量图如下，适合直接拖进 PPT：

- [`ppt_mappo_backbone_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_mappo_backbone_structure.svg)
  用途：第 5 页到第 7 页，讲 backbone 结构

- [`ppt_algorithm_evolution_timeline.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_algorithm_evolution_timeline.svg)
  用途：第 2 页、第 8 页、第 16 页，讲整体演化

- [`ppt_v1_v3_residual_family.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v1_v3_residual_family.svg)
  用途：第 9 页到第 11 页，讲 `v1-v3`

- [`ppt_v4_targeted_fusion_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v4_targeted_fusion_structure.svg)
  用途：第 12 页，讲 `v4`

- [`ppt_v5_dualstream_structure.svg`](/Users/zerick/code/MAIC/paper/figures/generated/ppt_v5_dualstream_structure.svg)
  用途：第 13 页到第 15 页，讲 `v5`

## 四、建议插入的现有实验图

现有图表里，最适合直接进 PPT 的有：

- [`join1_test_winrate_comparison.png`](/Users/zerick/code/MAIC/paper/figures/generated/join1_test_winrate_comparison.png)
  用于 `join1` 认证

- [`baseline_officialish_vs_warmstart.png`](/Users/zerick/code/MAIC/paper/figures/generated/baseline_officialish_vs_warmstart.png)
  用于 backbone 构建过程

- [`v1_v5_variant_summary.png`](/Users/zerick/code/MAIC/paper/figures/generated/v1_v5_variant_summary.png)
  用于版本对比总表

- [`v1_v5_representative_curves.png`](/Users/zerick/code/MAIC/paper/figures/generated/v1_v5_representative_curves.png)
  用于展示代表性训练曲线

- [`communication_diagnostics_attack.png`](/Users/zerick/code/MAIC/paper/figures/generated/communication_diagnostics_attack.png)
  用于攻击流诊断

- [`communication_diagnostics_move.png`](/Users/zerick/code/MAIC/paper/figures/generated/communication_diagnostics_move.png)
  用于移动流诊断

## 五、附录建议

附录建议保留以下内容：

- 附录 A：Backbone 关键超参数表
- 附录 B：`v1-v5` 版本结构变化总表
- 附录 C：各代表 run 的 `peak/final/last5`
- 附录 D：`v5` attack 与 move 指标解释
- 附录 E：当前后续实验计划

## 六、答辩时最值得强调的三个亮点

- 第一，不是直接堆通信模块，而是先在 `join1` 里验证通信机制本身是否可靠，再把它迁移到强 backbone 上。
- 第二，不是把 MAPPO 当成黑箱基线，而是系统地把它调成了一个可承载通信研究的 backbone。
- 第三，不是只汇报“哪个版本分高”，而是已经逐步抽出了机制性结论：语义、路由和融合位置三者缺一不可。

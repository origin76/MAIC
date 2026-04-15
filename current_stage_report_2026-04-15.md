# 现阶段实验报告（2026-04-15）

## 0. 数据口径与阅读说明

本报告用于总结当前阶段围绕 `5m_vs_6m` 与 `join1` 的 MAPPO / 通信机制实验进展，目标是为后续毕设写作提供一份可以直接扩写的中间版本。

本报告采用如下数据口径：

- 以 `results/sacred/*/cout.txt` 作为最终数值统计的主依据，优先读取其中的 `test_battle_won_mean`、`test_return_mean`、`test_ep_length_mean` 等字段。
- 以导出的 `results/...json` 作为绘制 TensorBoard 曲线和论文插图的主要数据源。
- 当 `cout.txt` 与终端临时输出不完全一致时，以 `cout.txt` 为准。
- 对于极少数最新 run，例如 `v5_dualstream_top1move_softplus`，`cout.txt` 中可能没有完整打印 `Finished Training`，但对应 `json` 已经正常导出，因此可视为训练基本完成，适合用于曲线绘制，但表格里的最终统计仍建议人工复核最后一个有效测试点。

本文中的三个常用指标定义如下：

- `peak`：测试胜率曲线中的峰值。
- `final`：最后一个有效测试点的测试胜率。
- `last5`：最后 5 个测试点评估胜率的平均值，用于衡量收尾稳定性。

---

## 1. Why MAPPO：为什么当前主线要回到 MAPPO

### 1.1 研究目标决定了必须先有一个可信的 Backbone

本课题的核心并不是“单纯证明通信有效”，而是要研究：

- 多智能体在 CTDE（集中式训练、分布式执行）范式下如何学习有效的信息交互；
- 信息应该如何编码；
- 应该和谁通信；
- 收到消息后应该如何融合；
- 现有协议为什么会失效，以及如何做机制层面的改进。

因此，实验主线必须先建立一个足够可信、可重复、且对通信改动敏感的基础策略网络。如果基础 MAPPO 自己都站不住，那么后续任何通信改造都很难解释：

- 胜率上不去，到底是通信模块不好，还是 Backbone 本身没学会；
- 曲线掉下去，到底是融合策略有问题，还是 PPO 更新本来就在发散；
- seed 方差很大，到底是多智能体协同难，还是基线实现本身不稳定。

这也是为什么实验推进到中后期之后，研究重点从“不断堆通信设计”转向了“先把 vanilla MAPPO 跑稳，再把通信小心地叠回去”。

### 1.2 MAPPO 适合作为本课题主干的原因

相较于值分解类算法或一开始就高度耦合通信结构的算法，MAPPO 作为主线有几方面优势：

- 它天然符合 CTDE 叙事。Actor 执行时只依赖局部信息，Critic 训练时可使用更强的全局或集中信息，和毕设主题高度一致。
- 它是当前多智能体强化学习中非常有代表性的 on-policy baseline，便于与官方工程和已有论文对齐。
- PPO 的约束更新形式比较适合后续做“微量通信注入”，即在不破坏原有策略主干的情况下，只向局部决策里加入小残差修正。
- 在 `5m_vs_6m` 这样的部分可观测协同微操任务中，RNN + centralized critic 的组合本身就有足够强的表达能力，适合先做强基线，再观察通信是否真能提供增益。

### 1.3 从实验上看，MAPPO 也是最值得继续投入的主线

当前阶段的实验已经给出一个非常明确的信号：

- `join1` 上，通信方法确实可以带来显著收益，但也非常容易彻底坍塌。
- `5m_vs_6m` 上，如果 Backbone 不够稳，通信模块会把问题放大，而不是自动解决问题。
- 一旦选定一个足够强的 MAPPO backbone，轻量、低增益、残差式的通信模块才开始表现出可解释的行为差异。

因此，当前主线选择是合理的：先固定一个可靠的 vanilla MAPPO Backbone，再逐步叠加“最小侵入”的通信适配器。

### 1.4 本节建议插图

建议增加一张总览图，题目可以命名为：

`图 1 课题主线框架：从 CTDE Backbone 到最小侵入通信模块`

图中建议展示：

- 左侧：局部观测输入 Actor；
- 中间：共享参数的 recurrent actor；
- 上方：训练期 centralized critic；
- 右侧：基础 local logits；
- 在 local logits 旁边额外画出“communication adapter”作为小残差支路，而不是替代主干。

这张图不需要直接引用日志，但建议在图注中明确说明：后续 v1-v5 的所有通信实验，都是建立在已训练好的 vanilla backbone 之上逐步演化而来。

---

## 2. Communication in `join1`：原型环境中的通信探索

### 2.1 为什么 `join1` 仍然值得保留在报告里

`join1` 不是最终主战场，但它在当前毕设叙事里仍然很重要，因为它承担了“通信原型验证环境”的角色。它适合用来回答两个早期问题：

- 通信机制是否有潜力在协作任务中产生明显增益；
- 如果通信机制设计不合理，是否会出现彻底坍塌。

也就是说，`join1` 更像是一个“协议试验台”，而不是最终 benchmark。

### 2.2 `join1` 上的关键现象

从已有 run 看，`join1` 的结论非常鲜明：

| 方法 | 代表 run | peak | final | last5 | 现象 |
| --- | --- | ---: | ---: | ---: | --- |
| `qmix` | `results/sacred/15/cout.txt` | 0.7400 | 0.5960 | 0.6704 | 可以学到协作，但不是最强 |
| `budgeted_sparse_mappo` | `results/sacred/19/cout.txt` | 1.0000 | 1.0000 | 1.0000 | 稀疏预算通信在该环境中极其有效 |
| `maic` | `results/sacred/9/cout.txt` | 0.0000 | 0.0000 | 0.0000 | 直接失败 |
| `maic_parallel` | `results/sacred/28/cout.txt` | 0.0000 | 0.0000 | 0.0000 | 并行版同样失败 |
| `maic_parallel_join1_tuned` | `results/sacred/29/cout.txt` | 0.0000 | 0.0000 | 0.0000 | 调参后仍未恢复 |

这里最有价值的不是“哪个方法数值最高”，而是它揭示出一个很重要的研究结论：

- 通信不是天然有用的；
- 通信协议、对象选择和融合方式如果设计不对，性能不只是“不提升”，而是可能彻底归零；
- 稀疏、受限、结构上更可控的通信反而更可能有效。

这为后续在 `5m_vs_6m` 上做“硬稀疏”“低带宽”“残差注入”等设计提供了非常直接的动机。

### 2.3 `join1` 在全文中的叙事定位

在最终论文里，建议把 `join1` 定位为：

- 早期通信验证环境；
- 用来揭示“通信机制本身也会失败”的现象；
- 为后续从“强表达通信”退回到“受约束、可解释、低风险通信”提供实验依据。

不要把 `join1` 写成最终结论环境，而应把它写成“暴露问题的显微镜”。

### 2.4 本节建议图表与引用日志

建议添加：

`图 2 join1 环境下不同方法的测试胜率曲线对比`

建议使用以下 JSON 绘制：

- `results/join1/qmix/2026-03-20_19-38-26_qmix_join1.json`
- `results/join1/budgeted_sparse_mappo/2026-04-02_16-39-47_budgeted_sparse_mappo_join1.json`
- `results/join1/maic_parallel/2026-04-02_20-15-25_maic_parallel_join1.json`
- `results/join1/maic_parallel_join1_tuned/2026-04-02_20-36-55_maic_parallel_join1_tuned_join1.json`

建议在正文或图注中同时引用以下日志作为最终统计来源：

- `results/sacred/15/cout.txt`
- `results/sacred/19/cout.txt`
- `results/sacred/28/cout.txt`
- `results/sacred/29/cout.txt`

---

## 3. MAPPO Baseline：从“能跑”到“可信”的基线主线

### 3.1 先做 vanilla baseline，是为了让后续通信实验可解释

`5m_vs_6m` 比 `join1` 更接近本文真正想讨论的问题：部分可观测、协作微操、局部动作和全局配合之间存在张力。也正因为如此，它比 `join1` 更难，通信设计稍微不慎就会把 Backbone 一起带崩。

当前实验的一个核心转折点就是：停止在不稳定的通信主线上盲目叠设计，转而先验证 vanilla MAPPO 在当前代码框架内能否跑出可信结果。

### 3.2 早期 officialish 版本说明“实现方向对了，但还不够稳”

较早的 `officialish` 版本虽然已经在实现层面向官方 MAPPO 靠拢，但其 seed 稳定性并不理想。代表性结果如下：

| 配置 | 代表 run | peak | final | last5 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| `vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor` | `results/sacred/75/cout.txt` | 0.6000 | 0.5750 | 0.4950 | 有学习能力，但收尾回落 |
| 同上 | `results/sacred/76/cout.txt` | 0.0875 | 0.0250 | 0.0250 | 明显 seed 敏感 |
| 同上 | `results/sacred/77/cout.txt` | 0.0250 | 0.0250 | 0.0150 | 近乎失效 |

这说明两个事实：

- 代码框架本身并非完全错误，否则 seed 1 不会学起来；
- 但优化过程仍然存在较强随机性，需要进一步引入更“官方式”的稳定化策略。

### 3.3 Warm-start 控制实验说明 Backbone 本身已经具备较强潜力

在选取较强 checkpoint 后进行 300k warm-start finetune，结果明显改善：

| 配置 | 代表 run | peak | final | last5 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| `finetune_control_300k_from_relaxactor1404757` | `results/sacred/81/cout.txt` | 0.7125 | 0.6500 | 0.6675 | 强 |
| 同上 | `results/sacred/82/cout.txt` | 0.7000 | 0.6500 | 0.6625 | 强 |
| 同上 | `results/sacred/83/cout.txt` | 0.6125 | 0.5750 | 0.5875 | 稳中略弱 |

这组结果非常关键，它证明了：

- 当前代码框架中的 vanilla MAPPO 并不是“先天跑不起来”；
- 一旦训练稳定，vanilla backbone 本身具备较强的微操能力；
- 因此后续通信设计完全可以建立在这一 backbone 上，而不是重新发明一个不稳定的大系统。

### 3.4 当前 backbone 的实验性结论

截至目前，可以将 `officialish + semistable + lr decay + kl stop + relax actor` 这条线视为当前的主 Backbone 来源。它的重要意义不在于“所有 seed 都最好”，而在于：

- 它是目前最接近官方 MAPPO 工程风格的实现；
- 它已经具备产出强 checkpoint 的能力；
- 它为后续所有 `microcomm` 系列提供了共同起点，保证后续版本之间的比较具有可解释性。

换言之，后续通信实验的关键不是重新训练一个全新的 Actor，而是在一个已经能打的 vanilla policy 周围，测试“极小通信支路是否能带来额外收益”。

### 3.5 本节建议图表与引用日志

建议添加：

`图 3 vanilla MAPPO backbone 的形成过程`

可以拆成两个子图：

- `图 3(a)`：`officialish relaxactor` 原始三 seed 曲线；
- `图 3(b)`：`warm-start control 300k` 三 seed 曲线。

建议引用的日志：

- `results/sacred/75/cout.txt`
- `results/sacred/76/cout.txt`
- `results/sacred/77/cout.txt`
- `results/sacred/81/cout.txt`
- `results/sacred/82/cout.txt`
- `results/sacred/83/cout.txt`

建议引用的 JSON：

- `results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor/2026-04-13_10-42-26_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor_sc2_5m_vs_6m.json`
- 同目录下另外两个 seed 的 JSON
- `results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_finetune_control_300k_from_relaxactor1404757/` 下三个 seed 的 JSON

---

## 4. v1-v5：通信模块的五个阶段

这一部分是当前实验最重要的“设计演化史”。建议在论文中将其明确写成一个逐步收敛的研究过程，而不是零散试错：

- v1：先证明最小通信残差支路可以挂在强 Backbone 上而不立即破坏策略；
- v2：尝试让注意力更锐化、通信更快收敛；
- v3：直接传“动作意图”，验证语义明确的信息是否更有效；
- v4：将通信严格限制为对攻击 logits 的小偏置修正，获得最清晰的机制解释；
- v5：进一步做双流解耦，区分“攻击协同信息”和“移动协同信息”，并揭示 move stream 的核心瓶颈。

### 4.1 v1：Microcomm Adapter 的最小侵入起点

v1 的核心思想是：不改 Backbone learner，不改 critic，不重写 PPO，只在 Actor 上加一个极小的通信适配器。

代表性结果如下：

| 版本 | 代表 run | peak | final | last5 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `v1_detach` | `results/sacred/79/cout.txt` | 0.6875 | 0.4750 | 0.5525 | 高峰值，但收尾回落 |
| `v1_end2end` seed1 | `results/sacred/80/cout.txt` | 0.6458 | 0.5250 | 0.5025 | 可学，但波动较大 |
| `v1_end2end` seed2 | `results/sacred/84/cout.txt` | 0.6250 | 0.4750 | 0.4875 | 一般 |
| `v1_end2end` seed3 | `results/sacred/85/cout.txt` | 0.6250 | 0.5625 | 0.5400 | 中等稳定 |

从 v1 可以学到的核心经验是：

- 低带宽、低增益、零初始化、残差式通信，确实能挂到强 Backbone 上而不立刻炸掉；
- `detach` 版更符合“固定 backbone”的实验设计原则；
- `end2end` 版虽然有更强的联合适应能力，但 seed 方差也更大。

### 4.2 v2：让通信更尖锐，但不要破坏主干

v2 主要围绕“注意力锐化”和“更积极地训练通信支路”展开，目的是让 adapter 不再长期停留在“平均听几个人但其实谁都没听清”的状态。

代表结果：

| 版本 | 代表 run | peak | final | last5 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `v2_sharp` | `results/sacred/86/cout.txt` | 0.6375 | 0.6375 | 0.6075 | 收尾较稳 |
| `v2_fastcomm` | `results/sacred/87/cout.txt` | 0.5833 | 0.5250 | 0.4625 | 提速不一定带来更好结果 |
| `v2_sharp_soft` seed1 | `results/sacred/88/cout.txt` | 0.5875 | 0.5875 | 0.5725 | 稳 |
| `v2_sharp_soft` seed2 | `results/sacred/89/cout.txt` | 0.6625 | 0.6000 | 0.6300 | 较好 |
| `v2_sharp_soft` seed3 | `results/sacred/90/cout.txt` | 0.6667 | 0.5250 | 0.4700 | 有回落 |

v2 的主要价值在于：它证明了“轻量通信”并不一定要追求非常复杂的语义设计，只要让注意力别太发散，性能就有机会接近或超过 v1 的平均水平。

### 4.3 v3：直接传动作意图，性能上来到当前第一梯队

v3 是当前整个通信线中最值得重视的一次跃迁。它不再传抽象 hidden feature，而是开始直接传“动作意图”这类物理语义更明确的信息。

代表结果：

| 版本 | 代表 run | peak | final | last5 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `v3_detach` | `results/sacred/91/cout.txt` | 0.6250 | 0.6250 | 0.5350 | 相对稳 |
| `v3_end2end` | `results/sacred/92/cout.txt` | 0.7000 | 0.6250 | 0.6275 | 强 |
| `v3_gatefloor` | `results/sacred/93/cout.txt` | 0.7250 | 0.5875 | 0.6750 | 当前 raw peak 最强代表 |
| `v3_gain` | `results/sacred/94/cout.txt` | 0.6875 | 0.5375 | 0.5850 | 有效但不如 gatefloor |
| `v3_fusionboost` | `results/sacred/95/cout.txt` | 0.6875 | 0.5250 | 0.5125 | 增益有限 |

v3 的实验结论很明确：

- 当通信内容具有非常明确的物理意义时，通信分支更容易学出作用；
- “我要打谁”这种动作意图，比抽象的隐状态特征更适合作为微操协同信号；
- 但 end-to-end 强融合虽然能把峰值推高，也更容易在收尾阶段出现回落。

如果从“纯性能”角度看，v3 是目前最强的一条通信线。

### 4.4 v4：Targeted Fusion，把通信严格限制在 attack logits 上

v4 的核心思想是当前阶段最有论文价值的一点：通信不要试图接管整个 Actor 隐状态，而只作为对攻击动作的一小段偏置修正。

其结构可以概括为：

- Backbone 正常产生 `local logits`；
- 通信分支只生成针对 `attack logits` 的 `delta`；
- 最终输出形式类似 `final_attack = local_attack + alpha * gate * delta_attack`；
- 这样做可以把“通信在做什么”解释得非常清楚：它不是在替代局部策略，而是在集火决策上做微调。

代表结果：

| 版本 | 代表 run | peak | final | last5 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `v4_base` | `results/sacred/96/cout.txt` | 0.6125 | 0.6125 | 0.5625 | 最干净、最可解释 |
| `v4_softuse` seed1 | `results/sacred/97/cout.txt` | 0.6375 | 0.5250 | 0.5825 | 中等 |
| `v4_softuse` seed2 | `results/sacred/98/cout.txt` | 0.7500 | 0.6000 | 0.6450 | 很高峰值 |
| `v4_softuse` seed3 | `results/sacred/99/cout.txt` | 0.7031 | 0.4000 | 0.4775 | 稳定性不足 |

v4 的结论可以概括为：

- `v4_base` 不是全场最高分，但它是当前最容易解释、最适合写进论文主方法图的一版；
- `softuse` 可以进一步提高峰值，但会把 seed 方差重新拉大；
- 因此 v4 在论文叙事中的价值，更多在于“机制正确性”和“融合策略解释性”，而不是绝对数值封顶。

### 4.5 v5：双流融合，开始真正区分“打谁”和“怎么走”

v5 的设计动机非常自然：既然攻击协同和移动协同可能对应不同类型的信息，那就不应该让它们共用一条模糊的通信流。

v5 因此拆成两路：

- `attack stream`：主要服务于集火；
- `move stream`：主要服务于走位或局部拓扑协调。

其意义并不只是“多加一路消息”，而是第一次显式尝试把战术意图和空间协调分开处理。

代表结果：

| 版本 | 代表 run | peak | final | last5 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `v5_base` | `results/sacred/100/cout.txt` | 0.6000 | 0.6000 | 0.5550 | 双流朴素版一般 |
| `v5_top1move` | `results/sacred/101/cout.txt` | 0.6750 | 0.5375 | 0.6150 | 硬稀疏 move stream 明显改善 |
| `v5_top1move_softplus` seed1 | `results/sacred/102/cout.txt` | 0.5625 | 0.5625 | 0.5475 | 稳但不高 |
| `v5_top1move_softplus` seed2 | `results/sacred/103/cout.txt` | 0.5875 | 0.5750 | 0.5775 | 稳但不高 |
| `v5_top1move_softplus` seed3 | `results/sacred/104/cout.txt` | 0.6000 | 0.5500 | 0.5300 | 稳但不高 |

v5 的科学价值非常高，尽管绝对性能未必超过 v3：

- 朴素双流并不自动更强，尤其 move stream 很容易退化成高熵平均；
- `top1move` 说明，移动流如果没有硬稀疏约束，就容易把局部结构信息均值化；
- 当 move attention 改成 top-1 后，性能明显改善，说明“谁的移动信息最值得听”本身就是一个需要强约束的问题；
- `softplus` 主要提升了跨 seed 的平滑性和下界，但没有显著提高上界，说明当前主要瓶颈不在 gate 激活函数本身，而在通信内容与融合位置。

### 4.6 v1-v5 的阶段性总结

如果将 v1-v5 放在一条统一主线上，可以得到如下结论：

1. 一开始最重要的是“通信不能破坏 Backbone”，因此 v1 的低带宽、零初始化、残差式设计非常关键。
2. 仅仅让 hidden-state 通信更快或更尖锐还不够，所以 v2 只带来了中等幅度改进。
3. 真正显著提升性能的是 v3，即将通信内容换成具有明确语义的动作意图。
4. 从论文方法的可解释性出发，v4 是当前最优雅的一版，因为它把通信作用限制在攻击子空间，叙事非常清楚。
5. 从机制分析角度看，v5 揭示了“移动协同信息”比“攻击协同信息”更容易发散，硬稀疏路由是必要的。

因此，当前可以给出一个很清晰的阶段结论：

- 若以“原始峰值表现”为目标，v3 最有竞争力；
- 若以“机制最清楚、最适合论文主方法”为目标，v4 最值得主推；
- 若以“发现下一阶段通信研究真正瓶颈”为目标，v5 的分析价值最高。

---

## 5. 当前总体认识：我们已经学到了什么

### 5.1 不要再把通信看成“大而全替代主干”的模块

现阶段最重要的经验之一是：在 `5m_vs_6m` 这样的任务里，通信如果直接拼接进大块隐状态、试图主导整个 Actor，往往会引入高熵噪声、破坏局部微操本能，并最终拖垮策略。

真正有效的方向反而更克制：

- 低带宽；
- 低增益；
- 残差注入；
- 只改局部子空间；
- 最好只在确实需要协同的动作子空间中起作用。

### 5.2 通信内容比通信公式更重要

从 v1 到 v5，越来越清楚的一点是：

- “怎么听”当然重要；
- 但“传什么”更重要。

抽象 hidden feature 很容易变成不知所云的平均向量，而“攻击意图”“目标偏置”“移动局部协调”这种带有明确物理意义的信息，更容易在稀疏奖励环境里学出来。

### 5.3 移动协同比攻击协同更难

当前实验已经很强地暗示：

- 集火协同的目标比较明确，因此 attack-intent 类消息比较容易有效；
- move coordination 则更依赖局部拓扑、距离关系和即时威胁，因此如果没有更强的结构约束，极易发散成高熵平均。

这也是为什么 v5 中 `top1move` 比朴素 move stream 更值得重视。

### 5.4 当前最合理的论文主叙事

如果从毕设全文角度组织，目前最顺的一条主叙事是：

1. 从 CTDE 和 vanilla MAPPO 出发，建立强 Backbone。
2. 通过 `join1` 说明通信既可能非常有效，也可能彻底失败。
3. 在强 Backbone 上引入最小侵入通信适配器，逐步验证：
   - 轻量残差通信可行；
   - 语义明确的信息更有效；
   - 攻击和移动应区别对待；
   - 硬稀疏约束对 move stream 尤其重要。
4. 最终提出一个“面向协作子空间、残差式、受约束”的通信融合观点。

这条叙事和毕设题目“理解信息交互协议设计思想，并针对信息交互效率、编码学习和融合策略做改进”的目标是高度对齐的。

---

## 6. 建议补充的图表、插图与日志引用清单

这一节建议在后续写论文时直接保留，作为“实验章素材清单”。

### 6.1 必做图表

#### 图 1：课题总览结构图

建议内容：

- CTDE 范式；
- centralized critic；
- decentralized recurrent actor；
- communication adapter 作为小残差支路插入。

用途：

- 放在方法总览或实验总览前面，帮助读者快速建立结构理解。

日志引用：

- 无需直接引用日志。

#### 图 2：`join1` 上不同通信方法的测试胜率曲线

建议内容：

- `qmix`
- `budgeted_sparse_mappo`
- `maic_parallel`
- `maic_parallel_join1_tuned`

建议 JSON：

- `results/join1/qmix/2026-03-20_19-38-26_qmix_join1.json`
- `results/join1/budgeted_sparse_mappo/2026-04-02_16-39-47_budgeted_sparse_mappo_join1.json`
- `results/join1/maic_parallel/2026-04-02_20-15-25_maic_parallel_join1.json`
- `results/join1/maic_parallel_join1_tuned/2026-04-02_20-36-55_maic_parallel_join1_tuned_join1.json`

建议对照日志：

- `results/sacred/15/cout.txt`
- `results/sacred/19/cout.txt`
- `results/sacred/28/cout.txt`
- `results/sacred/29/cout.txt`

#### 图 3：vanilla MAPPO baseline 形成过程

建议内容：

- `officialish relaxactor` 三 seed；
- `warm-start control 300k` 三 seed；
- 横轴 `t_env`，纵轴 `test_battle_won_mean`。

建议对照日志：

- `results/sacred/75/cout.txt`
- `results/sacred/76/cout.txt`
- `results/sacred/77/cout.txt`
- `results/sacred/81/cout.txt`
- `results/sacred/82/cout.txt`
- `results/sacred/83/cout.txt`

#### 图 4：v1-v5 版本总表

建议形式：

- 一个总表即可，不一定非要画曲线；
- 列为 `版本 / 代表配置 / 代表 run / peak / final / last5 / 核心结论`。

建议覆盖 run：

- v1：`79, 80, 84, 85`
- v2：`86, 87, 88, 89, 90`
- v3：`91, 92, 93, 94, 95`
- v4：`96, 97, 98, 99`
- v5：`100, 101, 102, 103, 104`

建议日志来源：

- `results/sacred/79/cout.txt` 到 `results/sacred/104/cout.txt`

#### 图 5：v4 与 v5 的通信机制诊断图

建议内容：

- `gate_mean`
- `no_comm_prob`
- `mean_attn_entropy`
- `message_norm`

推荐重点比较：

- `v4_base`：`results/sacred/96/cout.txt`
- `v5_base`：`results/sacred/100/cout.txt`
- `v5_top1move`：`results/sacred/101/cout.txt`
- `v5_top1move_softplus`：`results/sacred/102/cout.txt`、`103/cout.txt`、`104/cout.txt`

建议图意：

- 用于证明“move stream 的核心问题不是单纯 gate 太小，而是没有强结构约束时容易高熵平均”；
- `top1move` 的价值在于把 move attention 从“平均听大家”改成“明确听某一个邻居”。

### 6.2 建议补充的机制插图

#### 图 6：v4 targeted fusion 结构图

建议内容：

- local logits 分成 move / attack；
- communication branch 只生成 `attack delta`；
- 最终只修正 `attack logits`。

这张图很适合放在正文主方法部分，因为它是当前最干净、最容易解释的设计。

#### 图 7：v5 dual-stream 结构图

建议内容：

- 一条 attack stream；
- 一条 move stream；
- move stream 上特别标出 `top1` 稀疏选择；
- 最终分别修正 attack logits 和 move logits。

这张图的作用不只是展示结构，更重要的是体现“移动协同和攻击协同不是同一种信息”。

### 6.3 建议在正文中明确引用的关键日志

如果论文正文只想引用少量最关键日志，建议优先保留下面这些：

- `results/sacred/19/cout.txt`
  - 说明 `join1` 上受约束通信可以非常强。
- `results/sacred/75/cout.txt`
  - 说明 officialish baseline 的代表性学习曲线。
- `results/sacred/81/cout.txt`
  - 说明 warm-start backbone 的强性能上限。
- `results/sacred/93/cout.txt`
  - 说明 v3 action-intent + gate floor 的峰值能力。
- `results/sacred/96/cout.txt`
  - 说明 v4 targeted fusion 的可解释版本。
- `results/sacred/101/cout.txt`
  - 说明 v5 top1move 的机制突破点。

---

## 7. 现阶段结论与下一步建议

### 7.1 现阶段可以比较稳地写进报告的结论

当前阶段已经可以相对稳妥地写出以下结论：

- MAPPO 是本课题合理且必要的主 Backbone，因为只有先把 CTDE 下的 vanilla policy 跑稳，通信改进才有解释空间。
- `join1` 证明通信不是天然有效的，通信协议如果设计错误，性能会直接坍塌。
- 在 `5m_vs_6m` 上，最有效的通信不是“大而全”的 hidden-state 替代，而是低带宽、低增益、残差式、面向特定动作子空间的微量修正。
- 语义明确的消息比抽象隐层特征更有效，尤其是动作意图共享。
- 攻击协同比移动协同更容易学；移动协同需要更强的结构先验或硬稀疏约束。

### 7.2 下一步如果继续推进，建议优先做什么

如果后续还要继续做一轮实验，建议优先顺序如下：

1. 固定 strongest vanilla backbone，不再频繁改 PPO 主干。
2. 以 v4 为论文主方法骨架，因为它最好解释。
3. 将 v5 的发现作为机制扩展，重点写“为什么移动流必须硬稀疏”。
4. 若还要继续冲性能，再在 v4 或 v5 上做极小范围调参，而不要重新回到大规模、强耦合通信结构。

当前最值得珍惜的并不是某一个单独高点，而是已经逐渐形成了一条清晰、可讲述、可写进毕设的研究逻辑链条。

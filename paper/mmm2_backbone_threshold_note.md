# MMM2：Backbone 阈值与通信介入时机阶段总结

记录时间：2026-05-13

## 目的

本记录总结 `MMM2` 地图上的一组关键对照实验，回答两个问题：

1. 在异质高难地图上，当前双流轻量通信是否能优于纯 MAPPO backbone？
2. 如果可以，这种收益是否依赖于 backbone 先达到某个更高的能力阈值？

本组实验的价值不在于再证明一次 `5m_vs_6m` 主线，而在于澄清：

```text
MMM2 上通信不是“始终有效”或“始终无效”，
而是对介入时机高度敏感。
```

## 对照对象

### 1. 纯 backbone 基线

模型目录：

```text
results/models/2026-05-12_22-04-48_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_sc2_MMM2
```

日志：

```text
results/sacred/365/cout.txt
```

这是 `MMM2` 上的纯 MAPPO backbone 训练轨迹，用于选择后续通信热启动的起点。

### 2. 从 `476836` 起点热启动通信

通信模型目录：

```text
results/models/2026-05-13_11-38-20_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_mmm2_midwarm_476836_sc2_MMM2
```

日志：

```text
results/sacred/367/cout.txt
```

这是第一次出现明显正信号的 `MMM2` 通信线。

### 3. 从 `476836` 起点继续跑等预算 backbone

模型目录：

```text
results/models/2026-05-13_16-07-33_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_mmm2_equalbudget_476836_sc2_MMM2
```

日志：

```text
results/sacred/368/cout.txt
```

该对照用于回答：

```text
476836 -> v5b 的提升，
究竟来自通信，还是只是因为继续多跑了 500k env steps？
```

### 4. 从更强 backbone 点 `476449` 再次热启动通信

通信模型目录：

```text
results/models/2026-05-13_18-23-28_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_mmm2_midwarm_476449_sc2_MMM2
```

日志：

```text
results/sacred/369/cout.txt
```

这里的 `476449` 不是初始 backbone run 的点，而是 `368` 中更强 continuation backbone 的已保存 checkpoint。

### 5. 从同一 `476449` 起点继续跑等预算 backbone

模型目录：

```text
results/models/2026-05-13_21-01-46_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_mmm2_equalbudget_476449_sc2_MMM2
```

日志：

```text
results/sacred/370/cout.txt
```

该对照用于回答：

```text
如果 backbone 已经更强了，
通信是否终于能带来超出纯 continuation 的额外收益？
```

## 第一轮结论：`476836` 介入过早

在 `476836` 起点上：

- 通信线 `367` 最好点约为：

```text
t_env = 492940
test_return_mean = 10.3798
test_battle_won_mean = 0.0375
```

- 等预算 backbone 线 `368` 最好点约为：

```text
t_env = 493052
test_return_mean = 10.6022
test_battle_won_mean = 0.1125
```

因此第一轮公平对照的结论是：

```text
在 476836 这个起点，
当前通信热启动并未优于同预算的纯 backbone continuation。
```

这说明此前 `476836 -> v5b` 的正信号不能直接归因于通信本身，
至少在这个能力阶段，backbone 继续优化的收益更大。

## 第二轮结论：`476449` 是更合适的通信介入区间

从更强的 backbone 点 `476449` 出发后，结果发生了反转。

### 通信线 `369`

在 `369` 中观察到的最好点为：

```text
t_env = 432967
test_return_mean = 11.7212
test_battle_won_mean = 0.1312
```

附近还有一个相近高点：

```text
t_env ≈ 426k
test_return_mean = 11.6629
test_battle_won_mean = 0.1187
```

### 等预算 backbone 线 `370`

在 `370` 中观察到的最好点仅为：

```text
t_env = 473047
test_return_mean = 9.4884
test_battle_won_mean = 0.0063
```

到末段时甚至下降到：

```text
t_env = 493193
test_return_mean = 9.1132
test_battle_won_mean = 0.0000
```

因此第二轮公平对照的结论是：

```text
在 476449 这个更强 backbone 起点上，
通信热启动显著优于同预算的纯 backbone continuation。
```

## 机制层面的解释

这组结果最重要的意义是，它把 `MMM2` 上通信的失败与成功明确分成了两个阶段：

### 阶段 A：能力不足时过早介入

在 `476836` 起点，
backbone 自身仍有较大后续提升空间。
此时加入通信更像是在一个尚未成熟的局部策略面上叠加残差修正，
结果是通信没有提供足够高的决策杠杆，甚至打断了 backbone 自身的后续优化。

### 阶段 B：主干成熟后再介入

在 `476449` 起点，
backbone 已经形成了更强的异质协作基础。
这时通信模块不再需要“从零塑造协作”，
而是在一个更稳定的局部决策结构上提供动作级残差修正，
因此更容易转化为实际收益。

一句话概括：

```text
MMM2 上存在一个更高的 backbone competence threshold；
通信收益并非单调存在，而是需要在 backbone 达到较强协作能力后才更容易释放出来。
```

## 通信机制指标：`369` 峰值点是“真实工作”的

在 `369` 最佳点附近，通信指标并非退化或沉默：

```text
targeted_attack_gate_mean ≈ 0.1371
targeted_attack_effective_delta_abs_mean ≈ 0.0159
targeted_attack_no_comm_prob ≈ 0.0244
targeted_move_gate_mean ≈ 0.0428
targeted_attack_head_0_entropy ≈ 0.3973
```

这说明：

1. attack 通信门并未关闭，而是保持中等开启。
2. effective delta 不是零，说明通信确实进入了动作修正电路。
3. no-comm 没有重新膨胀，说明提升不是靠“回到沉默”实现的。
4. attention 熵也没有坍缩到完全硬选择，通信仍保留一定选择性。

因此这次 `369` 的性能抬升更像是：

```text
在更强 backbone 上，
attack residual leverage 终于进入了“有用但不过激”的工作区间。
```

## 当前最稳妥的结论

基于 `365-370` 的全部对照，目前最稳妥的表述不是：

```text
通信在 MMM2 上稳定优于 backbone。
```

而是：

```text
在 MMM2 这类异质高难地图上，
当前通信框架对 backbone 阶段高度敏感。
当 backbone 尚未达到足够强的协作能力时，
通信未必优于纯 continuation；
但当 backbone 已达到更高能力阈值后，
通信可以显著优于同预算的纯 backbone continuation。
```

## 对后续实验的启发

这组结果为后续跨地图实验提供了两个直接启发：

1. `2phase` 仍然是必要协议。
   在高难或异质地图上，不宜直接把通信从训练开始端到端打开。

2. warmstart 时机本身是核心变量。
   后续若切换到 `2c_vs_64zg`、`3s5z_vs_3s6z` 或其他异质地图，
   应优先先找 backbone 的“能力阈值窗口”，
   再决定通信在哪个 checkpoint 介入。

## 论文可直接引用的表述

适合写入论文实验分析的一段英文式摘要可以是：

```text
On the heterogeneous MMM2 map, communication gains are not monotonic with
training progress. When communication is introduced too early, it does not
outperform an equal-budget backbone continuation. However, once the backbone
reaches a stronger coordination regime, communication warmstart from that
checkpoint yields a clear advantage over the equal-budget backbone baseline.
This suggests that communication effectiveness depends on a backbone
competence threshold rather than arising uniformly throughout training.
```

若写成中文，可用：

```text
在 MMM2 异质地图上，通信收益并非随训练阶段单调存在。
当通信在 backbone 能力尚不足时过早介入，其表现不优于同预算的纯 backbone continuation；
但当 backbone 已进入更强协作区间后，再引入通信则能够显著优于同预算 backbone 基线。
这表明通信有效性依赖于 backbone 的能力阈值，而不是在整个训练过程中均匀释放。
```

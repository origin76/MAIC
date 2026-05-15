# 2c\_vs\_64zg：Equal-Budget Pair 对照结果记录

记录时间：2026-05-14

## 目的

本记录总结 `2c_vs_64zg` 地图上的一组公平对照实验，目标是回答：

```text
在同一个强 backbone checkpoint 上，
当前双流轻量通信热启动是否能够优于同预算的纯 backbone continuation？
```

与 `MMM2` 那组实验相比，这里的重点不是“通信介入时机是否过早”，而是：

```text
当 backbone 已处于一个较强协作区间时，
通信是否能够稳定提供额外收益？
```

## 实验对象

### 1. Backbone 主干 run

模型目录：

```text
results/models/2026-05-13_23-49-03_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_2c_vs_64zg_sc2_2c_vs_64zg
```

日志：

```text
results/sacred/371/cout.txt
```

该 run 训练到 `2.5M` 步左右，在后段形成较明显的平台区，可作为后续 warmstart 的 backbone 主干。

### 2. 从 `2463279` 热启动通信

模型目录：

```text
results/models/2026-05-14_10-58-32_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_2c_vs_64zg_midwarm_2463279_sc2_2c_vs_64zg
```

日志：

```text
results/sacred/374/cout.txt
```

配置为：

```text
vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04
```

### 3. 从同一 `2463279` 继续跑等预算 backbone

模型目录：

```text
results/models/2026-05-14_12-44-44_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_2c_vs_64zg_equalbudget_2463279_sc2_2c_vs_64zg
```

日志：

```text
results/sacred/375/cout.txt
```

该对照用于保证比较公平性：

```text
通信线与 backbone 线从同一 checkpoint 出发，
并继续训练相同的 500k env steps。
```

## 起点 backbone 是否足够强

`371` 中 `2463279` 附近已经处于一个较强平台：

```text
t_env = 2456552
test_battle_won_mean = 0.5437
test_return_mean = 19.3473
```

紧邻的保存点为：

```text
2463279
```

因此，这里不是“弱 backbone 上强行加通信”，而是：

```text
从一个已形成较强作战能力的主干继续出发，
比较通信 continuation 与纯 continuation。
```

## 核心对照结果

### 通信线 `374`

后段性能持续上升，关键点如下：

```text
t_env = 466731
test_battle_won_mean = 0.6750
test_return_mean = 19.8801

t_env = 476825
test_battle_won_mean = 0.7063
test_return_mean = 20.0095

t_env = 487073
test_battle_won_mean = 0.7375
test_return_mean = 20.0744

t_env = 497355
test_battle_won_mean = 0.7625
test_return_mean = 20.1492
```

也就是说，通信线不是短暂尖峰，而是在末段形成了一段连续抬升区间。

### 纯 backbone continuation `375`

同起点、同预算下，纯 backbone continuation 没有继续提升，反而整体回落：

```text
t_env = 352727
test_battle_won_mean = 0.5625
test_return_mean = 19.4775
```

这是该线能观察到的最好点之一。之后大多数时间落在：

```text
test_battle_won_mean ≈ 0.46 - 0.51
test_return_mean ≈ 18.7 - 18.9
```

末段例如：

```text
t_env = 498528
test_battle_won_mean = 0.5063
test_return_mean = 18.8988
```

## 主结论

这组对照可以非常明确地给出结论：

```text
在 2c_vs_64zg 上，
从强 backbone checkpoint 2463279 出发，
当前 v5b 通信热启动显著优于同预算的纯 backbone continuation。
```

如果只看最佳点对比：

```text
communication:
  0.7625 win / 20.1492 return

backbone continuation:
  0.5625 win / 19.4775 return
```

即使和更保守的 backbone 后段均值相比，通信线也仍保持明显优势。

## 机制指标：收益不是靠高频“翻动作”实现的

通信线 `374` 最强区间附近的指标大致为：

```text
targeted_attack_gate_mean ≈ 0.25 - 0.28
targeted_attack_effective_delta_abs_mean ≈ 0.012 - 0.014
targeted_attack_no_comm_prob ≈ 0.19
attack_peer_conflict_fused_follow_peer_rate ≈ 0.010 - 0.012
attack_peer_conflict_fused_stay_local_rate ≈ 0.969 - 0.972
targeted_move_gate_mean ≈ 0.052
```

这些指标说明：

1. attack 通信门是打开的，但不是激进饱和。
2. `effective_delta` 为非零，说明通信确实进入了动作修正电路。
3. `no_comm` 保持在 `~0.19`，说明该图上的通信是“选择性使用”，而不是全时灌入。
4. `fused_follow_peer_rate` 仍很低，绝大多数情况下 fused attack 仍保持 local-dominated。

因此，性能提升的机制更像是：

```text
通信通过低频、轻量、持续的动作分布修正稳定改善决策，
而不是通过高频 peer-target 覆盖直接翻转本地决策。
```

这点很重要，因为它说明当前方法在 `2c_vs_64zg` 上的成功并非依赖于“通信强行接管”，而更像是：

```text
在保留局部主导决策结构的前提下，
通信提供了有效的细粒度校正。
```

## 与 2c\_vs\_64zg backbone 起点的关系

起点 `2463279` 大约是：

```text
0.5437 win / 19.3473 return
```

后续两条线的走向截然不同：

- 纯 backbone continuation：总体没有继续放大起点优势，反而回落
- 通信 continuation：不仅守住了起点，而且继续推升到 `0.76 / 20.15`

因此，这次结果不能解释为“只是继续训练更久”，而更合理地解释为：

```text
在 2c_vs_64zg 上，
通信在强 backbone 基础上确实提供了额外的有效增益。
```

## 与 MMM2 的关系

`2c_vs_64zg` 与此前 `MMM2` 结果形成了一个很好的对照：

- `MMM2`：通信收益对 backbone competence threshold 更敏感，需要更谨慎地挑选介入时机
- `2c_vs_64zg`：在强 backbone 点上，通信 continuation 对同预算 backbone continuation 的优势更加直接、稳定

因此，这张图提供的是一种更“干净”的正例：

```text
在复杂但相对更可控的地图上，
当前轻量双流通信机制可以在公平预算对照下稳定优于纯 backbone continuation。
```

## 当前最稳妥的论文表述

适合直接写进论文正文的一段中文概括可以是：

```text
在 2c_vs_64zg 地图上，以 2463279 步的强 backbone checkpoint 为共同起点，
通信 continuation 与纯 backbone continuation 在相同追加训练预算下进行公平对照。
结果表明，通信线在末段持续提升，并达到 0.7625 的测试胜率和 20.1492 的平均回报，
显著优于纯 backbone continuation 的 0.5625 胜率和 19.4775 回报。
进一步结合通信指标可知，该收益并非来自高频 peer-target 覆盖，
而更多体现为在 local-dominated 决策结构上施加低频、轻量而持续的动作分布修正。
```

若写成英文式摘要，可用：

```text
On 2c_vs_64zg, starting from the same strong backbone checkpoint (2463279),
the communication continuation consistently outperforms the equal-budget
backbone continuation, reaching 0.7625 test win rate and 20.1492 return.
Importantly, this gain is not associated with frequent peer-target overrides;
instead, the communication module improves performance through low-frequency,
lightweight, and persistent action-level corrections on top of a
local-dominated policy.
```

## 下一步建议

基于这组结果，最值得继续的不是再做大规模调参，而是：

1. 对通信线峰值点做 isolation eval
2. 对峰值点做 peer-local diagnostic
3. 检查这张图上的增益是否更多来自 attack 流还是 move 流

优先候选 checkpoint 为：

```text
results/models/2026-05-14_10-58-32_..._2c_vs_64zg_midwarm_2463279_sc2_2c_vs_64zg/478933
```

因为它已经非常接近后段性能高点，同时保留了完整训练上下文。

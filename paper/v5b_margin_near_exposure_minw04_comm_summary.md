# V5b Margin-Near Exposure (`min_weight=0.4`) 通信参数总结

记录时间：2026-05-09

## 实验定位

本记录对应配置：

```text
vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04
```

v5b 是在 v5 `margin-near / local-uncertainty-aware fused exposure` 基础上的一个最小修正：

```text
attack_peer_conflict_fused_local_conf_max = 0.90
attack_peer_conflict_fused_local_uncertainty_min_weight: 0.2 -> 0.4
```

它的目的不是引入新机制，而是回答一个更窄的问题：

```text
v5 的 uncertainty-aware fused exposure 是否因为最小权重过低而导致 gate / exposure 不稳定？
如果把 fused exposure 的最小权重从 0.2 提到 0.4，
能否保住 margin-near 选择性，同时避免某些 seed 再次掉回低 exposure？
```

因此，v5b 的结论可以直接服务于后续论文叙事中的“通信机制调稳”部分。

## 配置中的核心通信参数

v5b 中和 attack communication 直接相关的关键超参为：

```text
attack_fusion_scale = 0.5
attack_gate_max = 0.6

attack_no_comm_score_penalty = 0.3
attack_no_comm_target = 0.30
attack_no_comm_target_loss_weight = 0.02

attack_entropy_target = 0.50
attack_entropy_target_loss_weight = 0.005
attack_entropy_upper_only = True

attack_peer_conflict_margin_leverage_loss_weight = 0.005
attack_peer_conflict_attack_only_margin_loss_weight = 0.02
attack_peer_conflict_margin = 0.03
attack_peer_conflict_margin_huber_beta = 0.02
attack_peer_conflict_peer_support_threshold = 0.70
attack_peer_conflict_use_real_comm_weight = True
attack_peer_conflict_fixed_denom = True

attack_peer_conflict_fused_local_conf_max = 0.90
attack_peer_conflict_fused_local_uncertainty_min_weight = 0.4
```

这些参数背后的职责划分是：

- `silencebudget` 部分负责不让 attack stream 通过 no-comm token 逃回沉默。
- `fusion_scale + gate_max` 决定 residual 能有多大执行期杠杆。
- `peer_conflict_*` loss 负责让通信优先介入 peer-local conflict 状态。
- `fused_local_conf_max + fused_local_uncertainty_min_weight` 决定 fused exposure 对低 local-confidence 样本有多偏置，以及这种偏置最多能强到什么程度。

## 三个 Seed 的通信结果

训练导出 JSON：

```text
results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04/
  2026-05-08_21-47-03_...json   seed 1
  2026-05-08_23-17-04_...json   seed 2
  2026-05-09_12-12-36_...json   seed 3
```

下面统计统一使用 `last5` 均值，除 `final_win / peak_win / peak_t` 外，尽量反映末段稳定状态。

| Seed | final win | last5 win | peak win | peak t_env | last5 return | gate last5 | no-comm last5 | attack entropy last5 | effective delta last5 | fused follow peer last5 | attack-only follow peer last5 | fused stay local last5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6250 | 0.5625 | 0.8125 | 120600 | 15.7075 | 0.5364 | 0.0106 | 0.4453 | 0.0930 | 0.0068 | 0.0146 | 0.9925 |
| 2 | 0.9375 | 0.7250 | 0.9375 | 482615 | 17.2901 | 0.4416 | 0.0069 | 0.4476 | 0.0740 | 0.0243 | 0.0400 | 0.9751 |
| 3 | 0.6250 | 0.6375 | 0.8750 | 180915 | 16.1788 | 0.5642 | 0.0075 | 0.4356 | 0.1077 | 0.0113 | 0.0278 | 0.9887 |
| Mean | 0.7292 | 0.6417 | 0.8750 | 261377 | 16.3921 | 0.5141 | 0.0083 | 0.4428 | 0.0916 | 0.0141 | 0.0275 | 0.9854 |

## 末段通信权重与选择性统计

v5b 不只是“有没有通信”，还要看它把 fused exposure 压向了哪些样本。

| Seed | raw conflict weight last5 | fused weight last5 | uncertainty weight last5 | weighted local conf last5 | local conf last5 | peer conf last5 | conflict rate last5 | fused logp delta last5 | attack-only logp delta last5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1099 | 0.0486 | 0.4822 | 0.8505 | 0.8925 | 0.9166 | 0.1965 | -2.1727 | -2.1169 |
| 2 | 0.0358 | 0.0163 | 0.4871 | 0.8206 | 0.9075 | 0.9102 | 0.1076 | -1.8683 | -1.8209 |
| 3 | 0.0501 | 0.0222 | 0.4722 | 0.8491 | 0.9361 | 0.9441 | 0.0939 | -2.6116 | -2.5536 |
| Mean | 0.0653 | 0.0290 | 0.4805 | 0.8401 | 0.9120 | 0.9236 | 0.1326 | -2.2175 | -2.1638 |

这张表说明：

- `fused_weight_last5 / raw_weight_last5 ~= 0.44`，说明 fused exposure 仍然被明显下采样，但比 v5 的 `~0.29` 更强。
- `weighted_local_conf_last5 < local_conf_last5`，说明 fused exposure 的确偏向了 local 更不确定的样本，但这种偏置没有 v5 那么强。
- `fused_logp_delta_last5` 与 `attack-only logp delta last5` 仍然显著为负，表示 peer target 虽被注入，但在绝大多数冲突状态下仍未越过 local top1 margin。

## 与 v5 的对照结论

v5 的主要问题是：

```text
seed 1 的 fused exposure 太弱，
gate_last5 = 0.2347,
effective_delta_last5 = 0.0404,
导致 residual 虽有方向性，但很难稳定进入最终动作电路。
```

v5b 对这个问题的修复是成功的。三 seed 的通信均值相较 v5 呈现如下变化：

| Metric | v5 mean | v5b mean | Change |
|---|---:|---:|---:|
| final win | 0.6042 | 0.7292 | +0.1250 |
| last5 win | 0.5750 | 0.6417 | +0.0667 |
| last5 return | 15.6708 | 16.3921 | +0.7213 |
| gate last5 | 0.4564 | 0.5141 | +0.0577 |
| effective delta last5 | 0.0751 | 0.0916 | +0.0165 |
| fused weight last5 | 0.0159 | 0.0290 | +0.0131 |
| uncertainty weight last5 | 0.3602 | 0.4805 | +0.1203 |

但 v5b 不是无代价地更好。它也把 v5 中更激进的 peer-follow 倾向压回去了一部分：

| Metric | v5 mean | v5b mean | Change |
|---|---:|---:|---:|
| fused follow peer last5 | 0.0248 | 0.0141 | -0.0107 |
| attack-only follow peer last5 | 0.0578 | 0.0275 | -0.0303 |
| fused stay local last5 | 0.9713 | 0.9854 | +0.0141 |

因此，v5b 相对 v5 的真实含义不是“通信更会纠正动作”，而是：

```text
通信更稳定地进入了动作电路，
但通信并没有更频繁地越过 local margin。
```

## 机制总结

v5b 的三 seed 结果可以概括为：

1. `silencebudget` 持续有效。`no_comm_last5 ~= 0.0083`，attack stream 没有回到沉默。

2. `fused exposure` 被重新拉稳。`gate_last5 ~= 0.5141`，`effective_delta_last5 ~= 0.0916`，基本回到 v4 的强 exposure 水平。

3. `uncertainty-aware weighting` 仍然在起作用。`weighted_local_conf_last5 ~= 0.8401 < local_conf_last5 ~= 0.9120`，说明 fused loss 的样本选择仍然偏向相对更不确定的 local states。

4. 但最终动作仍然强烈 local-dominated。`fused_follow_peer_last5 ~= 0.0141`，`fused_stay_local_last5 ~= 0.9854`，说明通信虽然进了电路，却仍然很少真正改变 fused attack target。

5. 性能改进更像“把通信调稳”而不是“让通信更聪明”。v5b 在最终胜率和回报上比 v5 更好，但这种提升并没有伴随显著更高的 fused peer-follow rate。

## 论文可直接引用的表述

适合写进论文正文或实验讨论的一段简洁说法是：

```text
Raising the uncertainty-weight floor from 0.2 to 0.4 stabilizes attack-side
communication exposure across seeds: the attack gate remains open
(gate_last5 ~= 0.51), effective residual magnitude returns to the strong-exposure
regime (effective_delta_last5 ~= 0.09), and no-comm remains suppressed
(no_comm_last5 ~= 0.008). However, the fused policy is still overwhelmingly
local-dominated (fused_follow_peer_last5 ~= 0.014, fused_stay_local_last5 ~= 0.985).
Thus v5b repairs exposure stability, but not the causal decision-change bottleneck.
```

如果想写得更偏“研究脉络”一点，可以用：

```text
v5b is the strongest communication-stability variant in this line:
it preserves the silence-budget fix, restores stable action-level exposure,
and improves 3-seed performance. Yet its gains come mainly from stabilizing
communication-to-action injection rather than from making communication
frequently override local attack decisions.
```

## 对下一步的启发

v5b 基本说明：

```text
min_weight = 0.2  太弱，导致某些 seed 的 exposure 打不开；
min_weight = 0.4  更稳，但会牺牲一部分 margin-near 选择性；
```

因此下一步不宜继续只围绕 `min_weight` 小步调参。更有价值的方向是：

- 从 `local_confidence` proxy 转向更直接的 `local margin` proxy，例如 top1-top2 margin 或 local-vs-peer margin。
- 或者进入 `bad-action-aware` / `error-aware` 条件，只在本地更可能出错的冲突样本上提高 fused exposure。

一句话版本：

```text
v5b solved the exposure-stability problem of v5,
but not the causal decision-change problem.
```

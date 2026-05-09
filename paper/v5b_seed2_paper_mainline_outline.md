# V5b Seed2 论文主线叙事与图表清单

记录时间：2026-05-09

## 为什么选 V5b Seed2 作为主线

当前实验线上，`v5b_margin_near_exposure_minw04` 的 seed2 是最适合作为论文主线展示的单个 checkpoint / single-seed trajectory。原因不是它“证明通信已经彻底解决了决策因果性”，而是它同时满足了三件事：

1. 通信参数日志是合理的，不是靠异常沉默、异常 gate 饱和、或异常 residual 爆炸换来的高分。
2. 通信 exposure 已经稳定进入动作电路，能支撑论文里关于“communication-to-action leverage”修复的机制叙事。
3. 它仍然保留一个诚实的边界：通信虽然稳定进入动作层，但并没有经常越过 local attack margin。因此论文可以同时讲清楚“做对了什么”和“还没做成什么”。

这比选一个更极端但机制不干净的点更适合作为论文主线。

## 核心叙事

建议论文主线围绕下面这条叙事链展开：

```text
top2 routing + silence budget
    -> no-comm escape is suppressed
    -> peer-conflict communication can remain active
    -> uncertainty-aware fused exposure stabilizes action-side injection
    -> attack communication achieves stable action-level leverage
    -> but fused decisions are still mostly local-dominated
```

因此，论文主线不应写成：

```text
we solved communication
```

而应写成：

```text
we identified and repaired the communication exposure bottleneck,
while also diagnosing the remaining causal decision-change bottleneck.
```

## V5b Seed2 的主线数值

主线 checkpoint：

```text
config:
vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04

json:
results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04/
2026-05-08_23-17-04_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_sc2_5m_vs_6m.json
```

推荐正文直接引用的关键数值：

| Metric | Value |
|---|---:|
| final win | 0.9375 |
| last5 win | 0.7250 |
| peak win | 0.9375 |
| peak t_env | 482615 |
| final return | 19.3679 |
| last5 return | 17.2901 |
| attack gate last5 | 0.4416 |
| attack no-comm last5 | 0.0069 |
| attack entropy last5 | 0.4476 |
| effective delta last5 | 0.0740 |
| fused follow peer last5 | 0.0243 |
| attack-only follow peer last5 | 0.0400 |
| fused stay local last5 | 0.9751 |
| raw conflict weight last5 | 0.0358 |
| fused weight last5 | 0.0163 |
| uncertainty weight last5 | 0.4871 |
| weighted local conf last5 | 0.8206 |
| local conf last5 | 0.9075 |
| peer conf last5 | 0.9102 |
| conflict rate last5 | 0.1076 |
| fused logp delta last5 | -1.8683 |

一句话读法：

```text
V5b seed2 is a high-performing, communication-active, non-silent, moderately-gated,
action-level exposure regime in which communication is clearly alive but still
rarely flips final attack decisions.
```

## 与相邻版本的主线对比

建议论文主线只保留最必要的邻近对照：`signsplit seed2 -> v4 seed2 -> v5 seed2 -> v5b seed2`。

### 1. Signsplit Seed2：通信存在，但几乎不影响动作

| Metric | Signsplit seed2 |
|---|---:|
| final win | 0.3125 |
| last5 win | 0.5875 |
| gate last5 | 0.0943 |
| effective delta last5 | 0.0013 |
| no-comm last5 | 0.0037 |

叙事作用：

```text
作为“通信已经不沉默，但基本碰不到动作”的基线。
```

### 2. V4 Seed2：通信 exposure 进入动作层

| Metric | V4 seed2 |
|---|---:|
| final win | 0.6875 |
| last5 win | 0.5125 |
| last5 return | 14.9991 |
| gate last5 | 0.5975 |
| effective delta last5 | 0.0917 |
| fused follow peer last5 | 0.0236 |
| attack-only follow peer last5 | 0.0297 |

叙事作用：

```text
第一次把 communication 从“几乎不碰动作”推进到“稳定进入 action logits”。
```

### 3. V5 Seed2：更强选择性，但仍有 exposure 波动风险

| Metric | V5 seed2 |
|---|---:|
| final win | 0.8750 |
| last5 win | 0.6875 |
| last5 return | 16.9514 |
| gate last5 | 0.5556 |
| effective delta last5 | 0.0911 |
| fused follow peer last5 | 0.0321 |
| attack-only follow peer last5 | 0.0603 |
| fused weight last5 | 0.0141 |
| uncertainty weight last5 | 0.3946 |

叙事作用：

```text
证明 uncertainty-aware fused exposure 能让通信更“聪明”，
但在其他 seed 上又会出现 exposure 不稳的问题。
```

### 4. V5b Seed2：主线版本

| Metric | V5b seed2 |
|---|---:|
| final win | 0.9375 |
| last5 win | 0.7250 |
| last5 return | 17.2901 |
| gate last5 | 0.4416 |
| effective delta last5 | 0.0740 |
| fused follow peer last5 | 0.0243 |
| attack-only follow peer last5 | 0.0400 |
| fused weight last5 | 0.0163 |
| uncertainty weight last5 | 0.4871 |

叙事作用：

```text
不是最激进的 peer-follow 版本，
而是 exposure 更稳、日志更合理、性能最好的一版。
```

## 论文里要讲清楚的边界

这是主线叙事里最重要的诚实边界，必须明确写出来：

### 已经做成的部分

```text
1. no-comm escape 被压住；
2. attack gate 保持在健康区间；
3. communication residual 稳定进入动作电路；
4. uncertainty-aware weighting 让 fused exposure 不再完全盲目施压；
5. 这些改动与更高的训练期 test win / return 共存。
```

### 还没有做成的部分

```text
1. fused policy 仍然极少跟随 peer target；
2. fused_stay_local_rate 仍接近 0.98；
3. fused_logp_delta 仍显著为负；
4. 因此通信还没有经常性地跨过 local decision margin；
5. 也就是说，communication-to-action leverage 被修复了，
   但 causal decision-change 还没有被彻底解决。
```

建议正文里直接给出类似表述：

```text
Our best stable variant does not show that communication frequently overrides local attack decisions.
Rather, it shows that communication can be kept active, non-silent, and stably injected into action logits
without destabilizing learning. The remaining bottleneck is not exposure, but decision-changing selectivity.
```

## 图表清单

下面是建议作为 paper 主线的图表清单。每张图都应只承担一个论点。

### Figure 1. 通信主线示意图

形式：

```text
Pipeline / mechanism diagram
```

内容：

```text
peer routing -> silence budget -> gated fusion -> action logits
```

要证明：

```text
论文的问题不是“有没有通信模块”，而是通信在哪一步失效。
```

### Figure 2. V5b Seed2 的 win / return 曲线

形式：

```text
single-seed line plot
```

内容：

- x 轴：`t_env`
- y 轴：`test_battle_won_mean`
- 可辅图或右轴：`test_return_mean`
- 标出 final point 和 peak point

要证明：

```text
v5b seed2 不是偶发尖峰，而是后段重新抬升并在终点达到最强点。
```

### Figure 3. 通信参数随训练变化的机制图

形式：

```text
multi-line plot
```

建议曲线：

- `targeted_attack_gate_mean`
- `targeted_attack_no_comm_prob`
- `targeted_attack_effective_delta_abs_mean`
- `attack_peer_conflict_fused_follow_peer_rate`

要证明：

```text
通信在 v5b seed2 上是稳定激活、非沉默、并进入动作层的。
```

### Figure 4. 邻近版本对照柱状图

形式：

```text
bar chart or grouped bar chart
```

对象：

- signsplit seed2
- v4 seed2
- v5 seed2
- v5b seed2

指标建议：

- `gate_last5`
- `effective_delta_last5`
- `fused_follow_peer_last5`
- `last5_win`

要证明：

```text
主线不是靠单一性能数字选出来的，
而是靠“稳定通信参数 + 合理性能 + 清楚的机制边界”。
```

### Figure 5. Peer-vs-Local 因果诊断图

形式：

```text
stacked bar / causal diagnostic summary
```

对象：

- signsplit baseline
- v4
- v5b

指标建议：

- `fused_follow_peer_on_conflict_rate`
- `fused_stay_local_on_conflict_rate`
- `attack_only_follow_peer_on_conflict_rate`

要证明：

```text
通信确实越来越“碰到动作”，
但 fused policy 仍然主要 stay local。
```

### Table 1. 主线配置表

内容：

- `attack_no_comm_score_penalty`
- `attack_no_comm_target`
- `attack_entropy_target`
- `attack_fusion_scale`
- `attack_gate_max`
- `attack_peer_conflict_margin_leverage_loss_weight`
- `attack_peer_conflict_attack_only_margin_loss_weight`
- `attack_peer_conflict_fused_local_conf_max`
- `attack_peer_conflict_fused_local_uncertainty_min_weight`

要证明：

```text
主线版本是一个非常小、可解释、可复现的修正，而不是大杂烩。
```

### Table 2. 主线通信指标表

建议直接使用：

- `final win`
- `last5 win`
- `final return`
- `gate last5`
- `no_comm last5`
- `effective_delta last5`
- `fused_follow_peer last5`
- `fused_stay_local last5`

要证明：

```text
通信参数是“合理的”，不是靠异常值换来性能。
```

## 正文建议写法

可以把主线结果压成两段：

### 主结果段

```text
Our selected mainline variant is v5b seed2, which combines the silence-budget
fix with uncertainty-aware fused exposure and achieves the strongest stable
performance on this line. The key communication statistics remain well-behaved:
attack no-comm probability stays near zero, the gate remains moderately open,
and effective residual magnitude stays in the strong-exposure regime.
```

### 边界段

```text
At the same time, causal diagnostics show that communication still rarely
overrides the local attack decision. Thus the contribution of v5b is not that
it fully solves decision-level coordination, but that it cleanly repairs the
communication exposure bottleneck while making the remaining decision-change
limitation explicit.
```

## 结论

如果论文主线选 `v5b seed2`，最稳妥的 framing 是：

```text
We repaired communication exposure and made communication parameters healthy and stable.
We did not yet fully solve causal decision change.
```

这条 framing 的好处是：

- 它和现有日志一致；
- 它和 causal diagnostic 一致；
- 它不会被后续 isolation / peer-local 检查轻易推翻；
- 它保留了论文的技术价值：我们不只是做了一个更高分的黑箱，
  而是把“通信为何失效、修到了哪里、还剩什么边界”讲清楚了。

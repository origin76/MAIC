# V5b Seed2 论文主线 Storyboard

记录时间：2026-05-09

## 1. 主线定位

当前 attack communication 主线中，最适合作为论文核心展示对象的不是“最激进地追求 peer follow”的版本，而是：

```text
vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04
```

其中最适合拿来做 single-run mainline 的轨迹是 `seed2`。选择它的理由不是“它终于证明通信已经解决了所有因果决策问题”，而是它同时满足三条对论文最重要的要求：

1. 通信参数健康。没有沉默逃逸，没有 gate 地板，也没有 residual 爆炸。
2. 机制链条完整。通信已经稳定进入动作电路，不再停留在“有路由但碰不到动作”的阶段。
3. 边界清楚。它仍然明确暴露出 causal decision-change 尚未彻底解决，因此非常适合作为一条诚实且可 defended 的论文主线。

换句话说，`v5b seed2` 不是“通信完全成功”的例子，而是“通信机制第一次变得合理、稳定、且可解释”的例子。

## 2. 论文主张应该是什么

这条主线最稳妥的 claim 不是：

```text
we solved communication-based coordination
```

而是：

```text
we repaired the communication exposure bottleneck,
stabilized action-level communication injection,
and made the remaining causal decision-change bottleneck explicit.
```

更具体一点，正文可以围绕下面这三句话展开：

1. `top2 selective routing + silence budget` 解决了 attack stream 的沉默逃逸问题。
2. `peer-conflict margin leverage + uncertainty-aware fused exposure` 让通信 residual 稳定进入 attack action logits。
3. 即便如此，最终 fused decision 仍然主要由 local attack policy 主导，因此通信的 causal decision-change 仍不充分。

这三句话比单纯讲胜率更有论文价值，因为它们把“系统曾经在哪里失效、现在修到哪里、剩下什么瓶颈”讲清楚了。

## 3. 为什么是 V5b Seed2，而不是别的点

### 3.1 它不是靠异常参数换来的高分

`v5b seed2` 的末段指标：

| Metric | Value |
|---|---:|
| final win | 0.9375 |
| last5 win | 0.7250 |
| peak win | 0.9375 |
| peak t_env | 482615 |
| final return | 19.3679 |
| last5 return | 17.2901 |
| gate last5 | 0.4416 |
| no-comm last5 | 0.0069 |
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
| attack-only logp delta last5 | -1.8209 |

这些数值说明：

- `no_comm ~= 0.007`，attack stream 没有回到沉默。
- `gate ~= 0.44`，通信不是地板态，也不是硬顶到上限。
- `effective_delta ~= 0.074`，通信已经真实进入动作电路。
- `entropy ~= 0.45`，attention 不是死亡也不是完全随机。

因此，这个结果适合支撑“通信参数合理”这条叙事。

### 3.2 它比前序版本更像“机制稳定”，而不是“局部激进”

相邻 `seed2` 对照：

| Variant | final win | last5 win | gate last5 | effective delta last5 | fused follow peer last5 | attack-only follow peer last5 |
|---|---:|---:|---:|---:|---:|---:|
| signsplit | 0.3125 | 0.5875 | 0.0943 | 0.0013 | 0.0000 | 0.0068 |
| v4 | 0.6875 | 0.5125 | 0.5975 | 0.0917 | 0.0236 | 0.0297 |
| v5 | 0.8750 | 0.6875 | 0.5556 | 0.0911 | 0.0321 | 0.0603 |
| v5b | 0.9375 | 0.7250 | 0.4416 | 0.0740 | 0.0243 | 0.0400 |

这组对照最重要的含义不是“v5b 的 peer follow 数字最大”，而是：

- `signsplit` 说明“路由还活着，但动作层几乎没有 leverage”。
- `v4` 说明“通信第一次稳定打进动作电路”。
- `v5` 更激进，但在跨 seed 上 exposure 稳定性仍不够理想。
- `v5b` 不是最激进的 decision-change 版本，但它把机制调到了最适合写论文的平衡点。

这个平衡点可以概括成一句话：

```text
not the strongest peer-follow pressure,
but the cleanest stable exposure regime.
```

### 3.3 它比后续“继续追因果改动”的版本更适合作为主线

`v5c_bad_action_decision_change` 的作用更像负结果边界，而不是新的主线。它表明：

- 继续增加 exposure / bad-action correction 压力，并不会自动把通信变成更好的 causal decision changer。
- 更强的通信作用力可能让 `gate` 和 `effective_delta` 更高，但 final performance 与 fused follow 反而变差。

因此，`v5c` 更适合写成：

```text
stronger exposure alone is not enough
```

而不是论文的 main result。

## 4. 主线叙事链

建议论文把 `v5b seed2` 组织成下面这条机制链：

```text
top2 selective attack routing
    -> silence budget suppresses no-comm escape
    -> peer-conflict leverage creates action-side pressure
    -> uncertainty-aware fused exposure keeps that pressure stable
    -> communication residual consistently reaches attack logits
    -> but fused attack decisions still rarely cross the local margin
```

这条链里的每一步都应该有单独的图表或指标支持，避免一张图承担太多逻辑。

## 5. 这条主线已经能讲清楚什么

### 5.1 “通信参数合理”

这里的“合理”不是一个模糊词，而是可以明确解释为：

1. `no_comm` 被持续压低，说明没有通过 silence token 逃逸。
2. `gate` 维持在中等偏开的区间，说明通信被使用，但没有靠硬饱和工作。
3. `effective_delta` 处于稳定正值，说明 communication residual 真正进入了动作层。
4. `entropy` 维持在健康区间，说明 attention 既不是塌缩为死头，也不是无差别扩散。
5. `weighted_local_conf < local_conf`，说明 fused exposure 的样本偏置确实在朝“local 更不确定”的状态倾斜。

### 5.2 “机制稳定”

这里的“稳定”也需要具体化。建议用下面的含义：

1. 不是只在单个时间点突然冲高，而是在后段保持了健康通信参数。
2. 不是只在训练小样本 test 上看起来有效，而是和 peer-local diagnostic 的机制判断一致。
3. 不是通过极端 gate 或极端 residual 获得分数，而是在 moderate-gate regime 里达到高性能。

### 5.3 “已经修好的瓶颈”

`v5b` 真正修好的，是之前最核心的 failure mode：

```text
peer signal exists
    but cannot stably influence final action logits
```

从 `signsplit -> v4 -> v5b` 的变化来看，这个瓶颈现在已经可以说被修到了：

```text
peer signal exists
    -> enters action logits with measurable effective magnitude
    -> survives gating and silence suppression
    -> coexists with strong task performance
```

## 6. 这条主线不能越界 claim 什么

这是整篇论文里最需要讲清楚的边界。

### 6.1 不能 claim “通信经常改变最终决策”

`v5b seed2` 的末段：

| Metric | Value |
|---|---:|
| fused follow peer last5 | 0.0243 |
| attack-only follow peer last5 | 0.0400 |
| fused stay local last5 | 0.9751 |
| fused logp delta last5 | -1.8683 |

这说明：

- 通信会进入动作层；
- 通信偶尔会推动 decision 朝 peer 方向移动；
- 但最终 fused decision 绝大部分时候依然 stay local。

因此不能写：

```text
communication reliably overrides local attack choices
```

### 6.2 不能 claim “高胜率主要来自通信纠正了错误动作”

更稳妥的说法是：

```text
the best checkpoint combines healthy communication statistics with strong task performance,
but the current diagnostics do not support the stronger claim that performance gains are mainly driven
by frequent communication-induced decision reversals.
```

也就是说，`v5b seed2` 适合支撑：

```text
healthy communication and strong performance can coexist
```

但不适合直接支撑：

```text
performance gains are caused by frequent peer-over-local correction
```

### 6.3 不能把“机制活着”偷换成“因果问题解决”

当前最诚实的边界句式应该是：

```text
We repaired communication-to-action exposure,
but we did not yet fully solve causal decision change.
```

这句话和现有训练日志、隔离评估、peer-local 诊断是完全一致的。

## 7. 图表清单

下面这份清单尽量让每张图只承担一个论点。

### Figure 1. Attack communication failure-to-repair pipeline

形式：

```text
mechanism diagram / pipeline figure
```

内容：

```text
peer intent
    -> routing
    -> no-comm suppression
    -> gating
    -> residual fusion
    -> attack logits
    -> final action
```

要证明：

```text
论文的核心问题不是“有没有通信模块”，
而是通信会在哪个环节失效。
```

### Figure 2. Mainline performance curve for V5b seed2

形式：

```text
single-run line plot
```

建议数据：

- `test_battle_won_mean`
- `test_return_mean`

要证明：

```text
v5b seed2 不是单点噪声，而是一条后段稳定抬升并在终点达到最好结果的轨迹。
```

### Figure 3. Mainline communication-health curve for V5b seed2

形式：

```text
multi-line mechanism plot
```

建议曲线：

- `targeted_attack_gate_mean`
- `targeted_attack_no_comm_prob`
- `targeted_attack_effective_delta_abs_mean`
- `targeted_attack_mean_attn_entropy`

要证明：

```text
通信在主线 checkpoint 上是非沉默、非地板、非爆炸、并且真实进入动作层的。
```

### Figure 4. Neighboring-variant comparison

形式：

```text
grouped bar chart
```

对象：

- `signsplit seed2`
- `v4 seed2`
- `v5 seed2`
- `v5b seed2`

建议指标：

- `last5 win`
- `gate last5`
- `effective delta last5`
- `fused follow peer last5`

要证明：

```text
主线的选择依据不是“某个单独性能最好”，
而是“通信参数最合理、机制最稳定、性能也足够强”。
```

### Figure 5. Peer-vs-local causal boundary figure

形式：

```text
stacked bar chart or paired bar chart
```

对象：

- `signsplit`
- `v4`
- `v5b`

建议指标：

- `attack_only_follow_peer_on_conflict_rate`
- `fused_follow_peer_on_conflict_rate`
- `fused_stay_local_on_conflict_rate`

要证明：

```text
通信确实越来越能碰到动作，
但 fused policy 仍然大多数时候保持 local-dominated。
```

### Figure 6. Claim boundary figure

形式：

```text
two-panel summary figure
```

Panel A:

- healthy communication metrics:
  `no_comm`, `gate`, `effective_delta`, `entropy`

Panel B:

- unsolved causal metrics:
  `fused_follow_peer`, `fused_stay_local`, `fused_logp_delta`

要证明：

```text
the system is healthy at the communication-parameter level,
yet still conservative at the final decision level.
```

这张图很适合放在 discussion 或 mechanism summary 的位置。

## 8. 表格清单

### Table 1. Mainline configuration table

建议列出：

- `attack_no_comm_score_penalty`
- `attack_no_comm_target`
- `attack_entropy_target`
- `attack_fusion_scale`
- `attack_gate_max`
- `attack_peer_conflict_margin_leverage_loss_weight`
- `attack_peer_conflict_attack_only_margin_loss_weight`
- `attack_peer_conflict_fused_local_conf_max`
- `attack_peer_conflict_fused_local_uncertainty_min_weight`

要表达的点：

```text
主线版本是小而明确的机制修正，不是无边界堆超参。
```

### Table 2. Mainline communication-health table

建议列出：

- `final win`
- `last5 win`
- `final return`
- `gate last5`
- `no_comm last5`
- `entropy last5`
- `effective_delta last5`
- `weighted_local_conf last5`
- `local_conf last5`

要表达的点：

```text
通信参数是健康而可解释的。
```

### Table 3. Causal boundary table

建议列出：

- `fused_follow_peer last5`
- `attack_only_follow_peer last5`
- `fused_stay_local last5`
- `fused_logp_delta last5`
- `attack-only logp delta last5`

要表达的点：

```text
通信已经进入动作层，但还没有频繁跨过 local decision margin。
```

### Table 4. Neighboring-variant narrative table

建议只保留最必要的 4 个版本：

- `signsplit`
- `v4`
- `v5`
- `v5b`

每行配一句 narrative tag：

- `signsplit`: routing alive, leverage absent
- `v4`: exposure restored
- `v5`: more selective, less stable
- `v5b`: stable exposure mainline

这个表很适合放到 appendix，也可以作为正文图的配套。

## 9. 结果段落可直接复用的写法

### 9.1 主结果段

```text
Our mainline checkpoint is v5b seed2, which combines silence-budgeted attack routing
with uncertainty-aware fused exposure. This checkpoint achieves the strongest stable
performance on the current line while keeping the communication statistics well-behaved:
attack no-comm probability remains near zero, the attack gate stays moderately open,
and effective residual magnitude remains in a strong but non-saturated regime.
```

### 9.2 机制段

```text
Relative to earlier variants, v5b no longer fails at the exposure stage.
Peer communication is not merely routed; it is stably injected into attack logits
with measurable effective magnitude. This makes v5b the cleanest example in our study
of communication-to-action leverage becoming operational rather than dormant.
```

### 9.3 边界段

```text
However, causal diagnostics show that the fused attack policy still rarely overrides
the local attack preference. In other words, v5b repairs the communication exposure
bottleneck but does not yet fully solve causal decision change. The remaining challenge
is not how to keep communication alive, but how to make it selectively decisive.
```

## 10. 图表制作优先级

如果只先做一版最小 paper pack，建议优先顺序如下：

1. `Figure 2`: `v5b seed2` win / return 曲线
2. `Figure 3`: `v5b seed2` 通信参数健康曲线
3. `Figure 4`: `signsplit -> v4 -> v5 -> v5b` 邻近版本对照
4. `Figure 5`: peer-vs-local causal boundary 图
5. `Table 2`: 主线通信参数表
6. `Table 3`: causal boundary 表

如果这些图和表先齐了，整篇论文最核心的 story 就已经站住了。

## 11. 一句话版结论

最适合放在内部写作备忘里的总结句是：

```text
V5b seed2 is the paper mainline because it gives us the cleanest stable communication regime:
the communication parameters are healthy, the mechanism is operational at the action level,
and the remaining limitation is explicit rather than hidden.
```

最适合放在论文正文里的总结句是：

```text
Our best stable variant demonstrates healthy and effective communication-to-action exposure,
while also revealing that causal decision change remains the unsolved bottleneck.
```

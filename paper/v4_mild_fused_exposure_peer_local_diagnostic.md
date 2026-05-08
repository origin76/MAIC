# V4 Mild Fused Exposure Peer-Local Diagnostic

记录时间：2026-05-08

## 实验定位

本记录对应配置：

```text
vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v4_mild_fused_exposure
```

v4 接在 `peer_conflict_margin_leverage_v3_scale_probe` 后面。v3 的核心结论是：

- 放大 attack residual scale 后，`attack_only` 路径已经能产生少量 target flip。
- 但 fused policy 仍然很少跟随 peer，因为 gate/exposure 不够。
- 下一个瓶颈不是 routing，也不是 silence，而是 communication residual 进入最终 action logits 的 exposure。

v4 因此保留 `silencebudget p=0.3`，继续使用 attack top2 selective routing，并加入一个很弱的 fused exposure loss：

```text
attack_fusion_scale = 0.5
attack_gate_max = 0.6
attack_peer_conflict_attack_only_margin_loss_weight = 0.02
attack_peer_conflict_margin_leverage_loss_weight = 0.005
```

目标不是一次性得到最终算法，而是验证这条 causal chain 是否被推进：

```text
peer intent exists
    -> no-comm is suppressed
    -> gate opens
    -> residual becomes effective
    -> fused attack target sometimes changes
```

## 诊断设置

诊断脚本：

```text
script/eval_peer_local_causal_diagnostic.sh
```

评估设置：

```text
env = sc2_5m_vs_6m_local
test_nepisode = 128
mode = peer-vs-local causal diagnostic
```

本记录包含用户已完成的两个 v4 diagnostic runs。seed 1 还没有在这份表里补跑，因此下面结论主要作为 v4 机制诊断，而不是完整 3-seed 统计。

### Seed 2 命令

```zsh
zsh /Users/zerick/code/MAIC/script/eval_peer_local_causal_diagnostic.sh \
  vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v4_mild_fused_exposure \
  results/models/2026-05-08_13-06-11_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v4_mild_fused_exposure_sc2_5m_vs_6m \
  301335 \
  sc2_5m_vs_6m_local \
  128 \
  2
```

### Seed 3 命令

```zsh
zsh /Users/zerick/code/MAIC/script/eval_peer_local_causal_diagnostic.sh \
  vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v4_mild_fused_exposure \
  results/models/2026-05-08_14-37-06_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v4_mild_fused_exposure_sc2_5m_vs_6m \
  476880 \
  sc2_5m_vs_6m_local \
  128 \
  3
```

## Raw Diagnostic Results

| Metric | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|
| load_step | 301335 | 476880 | - |
| test_battle_won_mean | 0.7500 | 0.5000 | 0.6250 |
| test_return_mean | 17.3927 | 14.7995 | 16.0961 |
| test_ep_length_mean | 49.0625 | 45.9062 | 47.4844 |
| test_return_std | 4.5658 | 5.3189 | 4.9424 |
| attack_valid_count | 10920 | 11672 | 11296 |
| chosen_attack_count | 8800 | 8332 | 8566 |
| chosen_attack_rate | 0.8059 | 0.7138 | 0.7599 |
| conflict_count | 1692 | 1584 | 1638 |
| conflict_rate | 0.1549 | 0.1357 | 0.1453 |
| peer_conflict_rate | 0.1368 | 0.1241 | 0.1305 |
| peer_valid_rate | 0.8829 | 0.9147 | 0.8988 |
| peer_top1_prob_mean | 0.8820 | 0.9160 | 0.8990 |
| local_top1_prob_mean | 0.8638 | 0.9109 | 0.8874 |
| fused_top1_prob_mean | 0.8578 | 0.9108 | 0.8843 |
| attack_only_top1_prob_mean | 0.8539 | 0.9102 | 0.8821 |
| local_agreement_rate | 0.8451 | 0.8643 | 0.8547 |
| no_comm_prob_mean | 0.0003 | 0.0002 | 0.0003 |
| gate_mean | 0.6000 | 0.5583 | 0.5792 |
| delta_abs_mean | 0.3155 | 0.3213 | 0.3184 |
| effective_delta_abs_mean | 0.0946 | 0.0897 | 0.0922 |
| attack_only_flip_rate | 0.0234 | 0.0075 | 0.0155 |
| attack_only_follow_peer_on_conflict_rate | 0.0426 | 0.0278 | 0.0352 |
| attack_only_stay_local_on_conflict_rate | 0.9527 | 0.9697 | 0.9612 |
| attack_only_other_on_conflict_rate | 0.0047 | 0.0025 | 0.0036 |
| fused_flip_rate | 0.0125 | 0.0048 | 0.0087 |
| fused_follow_peer_on_conflict_rate | 0.0189 | 0.0152 | 0.0171 |
| fused_stay_local_on_conflict_rate | 0.9787 | 0.9823 | 0.9805 |
| fused_other_on_conflict_rate | 0.0024 | 0.0025 | 0.0025 |
| chosen_match_local_rate | 0.9914 | 0.9957 | 0.9936 |
| chosen_match_fused_rate | 1.0000 | 1.0000 | 1.0000 |
| chosen_match_peer_rate | 0.8436 | 0.8584 | 0.8510 |
| chosen_match_attack_only_rate | 0.9945 | 0.9971 | 0.9958 |

## 与 Signsplit 诊断基线对比

Signsplit 诊断基线来自：

```text
paper/peer_vs_local_causal_diagnostic.md
```

| Metric | Signsplit mean | V4 diagnostic mean | Change |
|---|---:|---:|---:|
| no_comm_prob_mean | 0.0019 | 0.0003 | -0.0016 |
| gate_mean | 0.0728 | 0.5792 | +0.5064 |
| delta_abs_mean | 0.0824 | 0.3184 | +0.2360 |
| effective_delta_abs_mean | 0.0006 | 0.0922 | +0.0916 |
| attack_only_follow_peer_on_conflict_rate | 0.0023 | 0.0352 | +0.0329 |
| fused_follow_peer_on_conflict_rate | 0.0000 | 0.0171 | +0.0171 |
| fused_stay_local_on_conflict_rate | 1.0000 | 0.9805 | -0.0195 |

这个对比说明 v4 不是无效改动。它确实把通信从“完全无法改变动作”推进到了“小幅但可测量地改变动作”。

## Key Findings

1. Silence/no-comm 仍然被解决。`no_comm_prob_mean` 只有约 `0.0003`，说明 v4 没有回到沉默逃逸。

2. Gate exposure 已经基本打开。seed 2 的 `gate_mean = 0.6000` 达到 `attack_gate_max`，seed 3 也有 `0.5583`。继续单纯提高 fused exposure loss 很可能只是把 gate 更硬地顶到上限，而不是带来更聪明的通信介入。

3. Effective residual 已经不再是零。v4 的 `effective_delta_abs_mean ~= 0.0922`，相比 signsplit 的 `0.0006` 是数量级跃迁。这说明通信 residual 已经进入 action-logit 电路。

4. 但 fused action flip 仍然很少。`fused_follow_peer_on_conflict_rate` 只有 `0.0171`，也就是在 peer-local 冲突状态里，最终 fused policy 只有约 1.7% 会跟随 peer target。

5. Policy 仍然由 local target 主导。`fused_stay_local_on_conflict_rate ~= 0.9805`，`chosen_match_local_rate ~= 0.9936`，说明大多数时候通信虽然被注入，但没有跨过本地 top1 margin。

6. Seed 2 的高胜率不能主要归因于强通信因果性。seed 2 diagnostic win 达到 `0.7500`，但 fused follow peer 只有 `0.0189`。这更像是 local/backbone 状态较好，通信有轻微正贡献或伴随效应，而不是通信已经成为主要决策机制。

7. Seed 3 的低胜率也不是“通信过强破坏策略”。seed 3 的 gate 和 effective delta 都健康，但 fused follow peer 仍然只有 `0.0152`。它更像是本地策略状态较弱，同时通信仍未足够精准地纠正 local error。

## 机制结论

v4 的机制结论应该表述为：

```text
v4 moves the system from zero causal action impact to small but measurable action impact.
However, fused policy remains overwhelmingly local-dominated.
Therefore v4 repairs exposure, but not value-aligned correction.
```

更具体地说：

```text
peer intent exists and is valid
    -> no-comm is nearly eliminated
    -> gate opens to roughly 0.56-0.60
    -> effective residual reaches roughly 0.09
    -> attack-only can sometimes move toward peer
    -> fused policy still rarely crosses the local top1 margin
```

因此当前瓶颈已经不是：

```text
no communication
bad routing
dead gate
zero residual
```

而是：

```text
communication changes too few final actions,
and the changed actions are not yet clearly value-aligned.
```

## 对 V5 的启发

v5 不应该简单继续加大 `attack_peer_conflict_margin_leverage_loss_weight` 或继续提高 gate。v4 的 gate 已经接近 cap，effective delta 也足够可测。继续加压大概率会让模型在所有 conflict 样本上硬推 peer target，带来噪声和不稳定。

更合理的下一版是做 `margin-near / local-uncertainty-aware fused exposure`：

- Attack-only loss 继续保留较宽的 peer-conflict 学习信号，让 residual 学会“往 peer target 方向修正”。
- Fused exposure loss 只在 local decision 比较不确定、local top1 margin 较小、或者 peer 与 local 冲突且 peer 有足够置信度的样本上更强。
- Loss 分母继续用 base valid denom 或固定有效样本数，避免少数 selected conflict 样本暴君式主导梯度。
- 新增日志应显式记录 local margin / local uncertainty 分布，以及 fused flip 是否集中在 margin-near 样本。

一个最小 v5 方向：

```text
base_conflict_weight =
    conflict * peer_valid * real_comm_weight * peer_support_score

local_uncertainty_weight =
    min_weight + (1 - min_weight) * clamp((local_conf_max - local_conf) / local_conf_max, 0, 1)

attack_only_weight = base_conflict_weight
fused_weight = base_conflict_weight * local_uncertainty_weight
```

建议初始超参：

```text
attack_fusion_scale = 0.5
attack_gate_max = 0.6
attack_peer_conflict_attack_only_margin_loss_weight = 0.02
attack_peer_conflict_margin_leverage_loss_weight = 0.005
attack_peer_conflict_fused_local_conf_max = 0.90
attack_peer_conflict_fused_local_uncertainty_min_weight = 0.2
```

一句话版本：

```text
v4 proved communication can enter the action circuit; v5 should teach it when to cross the local margin.
```

# Advantage-Guided Leverage Huber 结果与通信因果性备忘

记录时间：2026-05-06

## 实验定位

本实验接在当前最稳主线 `v6_attack_top2_selective_silencebudget` 后面，目标不是继续调 attention routing 或 alignment sampling，而是直接检查 `communication-to-action leverage`：

- 如果攻击通信已经会路由、不会沉默，但 isolation eval 仍然显示通信对动作无边际贡献，那么可能是通信 residual 太弱，无法稳定改变 attack action logits。
- 因此加入 `advantage-guided attack leverage`，只在正优势攻击样本上鼓励通信融合后的策略比本地策略更支持实际采取且有正优势的攻击动作。
- Huber 版是 squared 版的修正：squared penalty 在 margin 附近梯度太弱，导致机制几乎不激活；Huber penalty 用更线性的局部梯度推动 residual 产生可测量的 log-prob 改变。

## 配置与关键超参

配置：

```text
src/config/algs/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber.yaml
```

核心设置：

```text
attack_no_comm_score_penalty = 0.3
attack_adv_leverage_loss_weight = 0.02
attack_adv_leverage_margin = 0.03
attack_adv_leverage_loss_mode = huber
attack_adv_leverage_huber_beta = 0.03
attack_adv_leverage_fixed_denom = true
attack_adv_leverage_use_real_comm_weight = true
```

实现约束：

- 辅助 loss 只通过 attack communication residual 回传梯度，不更新本地 policy logits。
- 对应实现中 local logits 使用 detach：`local_attack_logits.detach() + attack_residual_logits`。
- 这样该 loss 诊断的是“通信残差是否能改变动作倾向”，而不是让 backbone 本地策略替辅助 loss 背锅。

## 训练结果

训练 JSON：

```text
results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber/2026-05-05_22-42-32_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m.json
results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber/2026-05-06_00-06-33_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m.json
results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber/2026-05-06_01-23-50_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m.json
```

训练 Sacred run：

```text
results/sacred/303  seed 1
results/sacred/304  seed 2
results/sacred/305  seed 3
```

最终 checkpoint：

```text
seed 1: results/models/2026-05-05_22-42-32_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m/476785
seed 2: results/models/2026-05-06_00-06-33_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m/476926
seed 3: results/models/2026-05-06_01-23-50_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m/476667
```

训练统计：

| Seed | final win | last5 | last10 | peak |
|---:|---:|---:|---:|---:|
| 1 | 0.7500 | 0.7500 | 0.6937 | 0.8125 @ 321249 |
| 2 | 0.6250 | 0.6375 | 0.5750 | 0.8750 @ 241198 |
| 3 | 0.6875 | 0.4625 | 0.4500 | 0.8125 @ 221589 |
| Mean | 0.6875 | 0.6167 | 0.5729 | 0.8333 |

机制指标末点均值：

| Metric | Mean |
|---|---:|
| `targeted_attack_gate_mean` | 0.170961 |
| `targeted_attack_no_comm_prob` | 0.014580 |
| `targeted_attack_mean_attn_entropy` | 0.431122 |
| `targeted_attack_delta_abs_mean` | 0.242313 |
| `targeted_attack_effective_delta_abs_mean` | 0.004949 |
| `attack_adv_leverage_loss` | 0.000116 |
| `attack_adv_leverage_logp_delta_mean` | 0.000500 |
| `attack_adv_leverage_weighted_logp_delta_mean` | 0.000574 |
| `attack_adv_leverage_hinge_mean` | 0.029515 |
| `attack_adv_leverage_weight_mean` | 0.396754 |
| `attack_adv_leverage_positive_ratio` | 0.505394 |
| `attack_adv_leverage_real_comm_mass` | 0.991530 |
| `approx_kl` | 0.001383 |
| `targeted_move_no_comm_prob` | 0.666866 |
| `targeted_move_mean_attn_entropy` | 0.512205 |

## 与相邻诊断版本对比

| Config | final | last5 | last10 | peak | attack gate | attack no-comm | attack entropy | effective delta | logp delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| silencebudget | 0.6667 | 0.6292 | 0.5812 | 0.8333 | 0.0708 | 0.0038 | 0.4485 | n/a | n/a |
| adv squared | 0.6042 | 0.5875 | 0.5875 | 0.8125 | 0.0670 | 0.0141 | 0.4417 | 0.000435 | 0.000012 |
| adv huber | 0.6875 | 0.6167 | 0.5729 | 0.8333 | 0.1710 | 0.0146 | 0.4311 | 0.004949 | 0.000500 |
| leverage floor | 0.4792 | 0.5583 | 0.5854 | 0.8125 | 0.1215 | 0.0028 | 0.4430 | 0.001214 | n/a |
| soft conflict-align | 0.6875 | 0.5708 | 0.5833 | 0.8333 | 0.0697 | 0.0111 | 0.4421 | n/a | n/a |

结论：

- Huber 版确实激活了 leverage 机制：gate、delta、effective delta、logp delta 都明显上升。
- 它没有破坏 silence budget 的关键结构：attack no-comm 仍然约 1.5%，attack entropy 仍在 0.43 附近。
- 训练 final win 均值略高于 silencebudget，但 last10 稳定性没有提升，而且 seed 3 后期明显回落。
- 这说明 Huber leverage 可以制造“通信改变动作 logits”的现象，但训练曲线本身不足以证明通信有执行期因果收益。

## Isolation eval

评估脚本：

```text
script/eval_attack_comm_all.sh
```

评估设置：

```text
3 seeds x 4 modes
test_nepisode = 128
normal / gate_open / gate_closed / no_attack
```

Sacred eval run 与日志：

| Run | Seed | Mode | Load step | `cout.txt` |
|---:|---:|---|---:|---|
| 306 | 1 | normal | 476785 | `results/sacred/306/cout.txt` |
| 307 | 1 | gate_open | 476785 | `results/sacred/307/cout.txt` |
| 308 | 1 | gate_closed | 476785 | `results/sacred/308/cout.txt` |
| 309 | 1 | no_attack | 476785 | `results/sacred/309/cout.txt` |
| 310 | 2 | normal | 476926 | `results/sacred/310/cout.txt` |
| 311 | 2 | gate_open | 476926 | `results/sacred/311/cout.txt` |
| 312 | 2 | gate_closed | 476926 | `results/sacred/312/cout.txt` |
| 313 | 2 | no_attack | 476926 | `results/sacred/313/cout.txt` |
| 314 | 3 | normal | 476667 | `results/sacred/314/cout.txt` |
| 315 | 3 | gate_open | 476667 | `results/sacred/315/cout.txt` |
| 316 | 3 | gate_closed | 476667 | `results/sacred/316/cout.txt` |
| 317 | 3 | no_attack | 476667 | `results/sacred/317/cout.txt` |

Isolation eval 结果：

| Seed | normal | gate_open | gate_closed | no_attack | normal - no_attack | gate_open - normal |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3125 | 0.3438 | 0.3125 | 0.3125 | 0.0000 | +0.0313 |
| 2 | 0.5859 | 0.5859 | 0.5938 | 0.5938 | -0.0079 | 0.0000 |
| 3 | 0.4141 | 0.4219 | 0.4219 | 0.4219 | -0.0078 | +0.0078 |
| Mean | 0.4375 | 0.4505 | 0.4427 | 0.4427 | -0.0052 | +0.0130 |

Return 均值：

| Mode | Mean test return |
|---|---:|
| normal | 14.2141 |
| gate_open | 14.3805 |
| gate_closed | 14.2457 |
| no_attack | 14.2457 |

关键观察：

- `gate_closed == no_attack` 在三个 seed 上完全一致，说明 eval 开关链路是可信的。
- `normal - no_attack` 均值为 -0.52%，没有稳定正向执行期攻击通信贡献。
- `gate_open - normal` 均值为 +1.30%，但主要来自 seed 1 的 +3.13%，幅度小且不稳定。
- 训练期 16 局 test final 明显高于 128 局 isolation normal，尤其 seed 1 和 seed 3。这说明小样本 test 会高估末点表现。

## 结论

这次实验给出的不是“advantage leverage 有效”，而是一个更细的边界结论：

```text
adv_leverage_huber 能让通信 residual 更强地进入 attack action logits，
但这种 action-level leverage 没有转化为稳定的 execution-time causal benefit。
```

因此：

- 不建议继续简单提高 `attack_adv_leverage_loss_weight` 或继续放大 residual/gate。
- 失败不在 routing：no-comm 被压住，attention entropy 健康。
- 失败也不只是强度：Huber 已经把 effective delta 提高到可测量水平。
- 真正缺的是“什么时候通信应该介入、介入方向是否纠正了本地策略错误”的因果选择机制。

## 对下一步通信因果性的启发

后续应该从“让通信更强”转向“让通信只在有因果价值的状态介入”。更值得检查的对象包括：

- 本地策略高不确定时，通信是否降低错误攻击概率。
- 本地目标与队友高置信目标冲突时，通信是否纠正目标选择。
- 通信打开前后的 action flip 是否集中在最终获胜或高优势片段。
- `normal - no_attack` 的收益是否只存在于少数 conflict/error states，而被全局平均稀释。
- 能否设计 state-conditioned causal eval，而不是只看整局平均胜率。

一句话版本：

```text
Huber leverage 证明“通信可以碰到动作”，但还没有证明“通信碰到的是该碰的动作”。
```

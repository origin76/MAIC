# Communication Cost Comparison Framework

记录时间：2026-05-09

## 目的

这份笔记用于给论文准备一个“可解释、可落表、可和仓库内实现直接对应”的通信成本对比框架。当前先完成两件事：

1. 明确一个统一的通信成本口径。
2. 先把本文当前最强主线与本仓库内 `maic` 实现做第一轮粗略对比。

这里的目标不是给出一个“绝对精确的比特级带宽统计”，而是给出一个在论文中足够清楚、并且能跨方法解释“谁在用更大执行期通信预算”的统一 proxy。

## 统一口径：执行期期望通信维数

本文建议把通信成本定义为每个 agent-step 的期望执行期通信维数：

\[
C_{\mathrm{exec}} = p_{\mathrm{comm}} \cdot k \cdot d_{\mathrm{exec}}.
\]

其中：

- `p_comm`：当前 step 非沉默通信的概率。若方法没有显式 no-comm token 或 hard gate，则按 `1.0` 处理。
- `k`：每步实际激活的接收者数。对 dense all-to-all 通信，`k = N - 1`。
- `d_exec`：每条 sender-receiver 边在执行期实际传入决策支路的消息维数。

这个定义强调的是“执行时到底有多少维信息被送进了别人策略”，而不是训练图中总共有多少内部隐变量。因此：

- attention query/key 维度不计入 `d_exec`；
- private latent sampling 维度若不直接作为执行期通信载体，也不计入 `d_exec`；
- 只有真正加到 peer 决策上的 message / residual / action-bias 才计入。

## 两种归一化方式

为了避免“只报一个数字但没有参照物”的问题，建议同时保留两种归一化。

### 1. 同消息维度下的拓扑归一化

\[
\rho_{\mathrm{topo}} = \frac{C_{\mathrm{exec}}}{(N-1)\cdot d_{\mathrm{exec}}}
= \frac{p_{\mathrm{comm}}\cdot k}{N-1}.
\]

它回答的问题是：

```text
在给定单边消息维度 d_exec 的前提下，
这个方法是否通过 top-k / silence / gate 真正减少了激活边数？
```

对于 dense all-to-all 方法，`\rho_topo = 1`。

### 2. 相对 64 维稠密隐藏态广播的工程归一化

\[
\rho_{64} = \frac{C_{\mathrm{exec}}}{(N-1)\cdot 64}.
\]

这对应一个很直观的工程参照：

```text
如果每个 agent 都向全部队友广播一个 64 维 hidden state，
当前方法的执行期通信成本相当于它的多少比例？
```

这个口径不是唯一正确答案，但在本仓库中很自然，因为：

- 我们当前 MAPPO 主干的 `rnn_hidden_dim = 64`；
- 很多“直接广播隐藏状态”的通信方法，在实现直觉上都接近这一密度。

## 当前已知的环境常数

下面的数值先固定在当前论文主线使用的 `SMAC 5m_vs_6m` 环境上。

- 友军智能体数：`N = 5`
- 因此每个 sender 最多可联系的队友数：`N - 1 = 4`
- 本仓库中攻击语义偏移量 `semantic_action_offset = 6`
- 在 `5m_vs_6m` 中敌人数为 `6`

据此可推得当前 SMAC 动作空间为：

\[
n_{\mathrm{actions}} = 6 + 6 = 12.
\]

这里的 `12` 不是从 `maic.yaml` 直接写死出来的，而是根据 SMAC 的标准动作构成和本仓库的语义动作切分推得的。若后续正文里想更严谨，可在实验设置节再补一句“SMAC attack actions equal the number of visible enemy slots”.

## 先和仓库内 MAIC 对比

### 我们当前 paper mainline：`v5b seed2`

对应配置：

[`src/config/algs/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04.yaml`](/Users/zerick/code/MAIC/src/config/algs/vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04.yaml)

对攻击通信流，当前可直接确定：

- `comm_value_dim = 8`
- `comm_topk = 2`
- `use_no_comm_token = True`
- `v5b seed2` 末段观测到 `attack_no_comm_prob ≈ 0.0069`

因此：

\[
p_{\mathrm{comm}} \approx 1 - 0.0069 = 0.9931,
\]
\[
k = 2,\quad d_{\mathrm{exec}} = 8,
\]
\[
C_{\mathrm{exec}}^{\mathrm{ours,attack}}
= 0.9931 \times 2 \times 8
\approx 15.9.
\]

进一步得到：

\[
\rho_{\mathrm{topo}}^{\mathrm{ours,attack}}
= \frac{0.9931\times 2}{4}
\approx 0.4966,
\]

\[
\rho_{64}^{\mathrm{ours,attack}}
= \frac{15.9}{4\times 64}
\approx 0.0621.
\]

解释上就是：

- 在“同样 8 维每边消息”的条件下，我们只激活了约 `49.7%` 的潜在边；
- 若和“向全部 4 个队友广播 64 维 hidden state”的稠密方案相比，当前攻击流执行期成本约为其 `6.2%`。

### 仓库内 `maic`

对应配置与实现：

- [`src/config/algs/maic.yaml`](/Users/zerick/code/MAIC/src/config/algs/maic.yaml)
- [`src/modules/agents/maic_agent.py`](/Users/zerick/code/MAIC/src/modules/agents/maic_agent.py)

这里最重要的区分是：

- `latent_dim = 8` 是 MAIC 的内部 latent 建模维度；
- 但执行期真正送进 peer 决策的是 `msg_net(...)-> n_actions` 产生的动作偏置向量。

从实现上可以直接看到：

- `msg` 的形状为 `(bs, n_agents, n_agents, n_actions)`
- attention `alpha` 的形状为 `(bs, n_agents, n_agents, 1)`
- 最终返回值是 `q + sum(alpha * msg, dim=1)`

因此 MAIC 在执行期最合适的通信维度口径不是 `latent_dim = 8`，而是：

\[
d_{\mathrm{exec}}^{\mathrm{maic}} = n_{\mathrm{actions}} = 12.
\]

同时它没有：

- top-k 稀疏选择；
- no-comm token；
- execution-time silence budget。

因此对当前 `5m_vs_6m`：

\[
p_{\mathrm{comm}} = 1,\quad k = N-1 = 4,\quad d_{\mathrm{exec}} = 12,
\]
\[
C_{\mathrm{exec}}^{\mathrm{maic}} = 1\times 4 \times 12 = 48.
\]

进一步得到：

\[
\rho_{\mathrm{topo}}^{\mathrm{maic}} = 1,
\]
\[
\rho_{64}^{\mathrm{maic}} = \frac{48}{4\times 64} = 0.1875.
\]

因此，在“每个 agent-step 的执行期通信维数”这一口径下：

- 我们当前攻击流主线：`15.9`
- 仓库内 MAIC：`48`

也就是：

\[
\frac{15.9}{48} \approx 33.1\%.
\]

换言之，当前 `v5b seed2` 攻击通信流的执行期通信成本大约是仓库内 MAIC 的三分之一。

## 第一版表格

### Markdown 版

| Method | Routing topology | `p_comm` | `k` | `d_exec` | `C_exec` | `rho_topo` | `rho_64` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Ours `v5b seed2` attack stream | top-2 selective + no-comm | 0.9931 | 2 | 8 | 15.9 | 0.4966 | 0.0621 | Attack-only stream; `attack_no_comm_prob ≈ 0.0069` |
| Repo `maic` | dense all-to-all targeted action bias | 1.0 | 4 | 12 | 48.0 | 1.0000 | 0.1875 | `latent_dim=8` is internal; execution message is `n_actions=12` |
| `budgeted_sparse_mappo_sc2_5m6m_sparse_stable` | top-3 sparse comm | 1.0 | 3 | 4 | 12.0 | 0.7500 | 0.0469 | No explicit no-comm token; sparse but always on |
| Dense hidden-state broadcast (64d ref.) | dense all-to-all | 1.0 | 4 | 64 | 256.0 | 1.0000 | 1.0000 | Reference row for engineering intuition |

### LaTeX 版

```tex
\begin{table}[t]
\centering
\caption{Execution-time communication cost comparison on SMAC 5m\_vs\_6m.
$C_{\mathrm{exec}} = p_{\mathrm{comm}} \cdot k \cdot d_{\mathrm{exec}}$ measures
the expected communicated dimensions per agent-step.}
\label{tab:comm_cost_compare}
\begin{tabular}{lccccccc}
\toprule
Method & Topology & $p_{\mathrm{comm}}$ & $k$ & $d_{\mathrm{exec}}$ & $C_{\mathrm{exec}}$ & $\rho_{\mathrm{topo}}$ & $\rho_{64}$ \\
\midrule
Ours v5b seed2 (attack) & top-2 + no-comm & 0.9931 & 2 & 8  & 15.9 & 0.4966 & 0.0621 \\
Repo MAIC              & dense all-to-all & 1.0000 & 4 & 12 & 48.0 & 1.0000 & 0.1875 \\
Sparse MAPPO stable    & top-3 sparse     & 1.0000 & 3 & 4  & 12.0 & 0.7500 & 0.0469 \\
Dense 64d broadcast    & dense all-to-all & 1.0000 & 4 & 64 & 256.0 & 1.0000 & 1.0000 \\
\bottomrule
\end{tabular}
\end{table}
```

## 这张表当前能支撑什么结论

在当前可确定的信息下，这张表已经足以支撑三个相对稳妥的论文表述。

### 1. 本文 strongest mainline 不是靠大带宽通信取胜

从 `v5b seed2` 的攻击流看，执行期期望通信成本只有：

\[
C_{\mathrm{exec}} \approx 15.9.
\]

相对 64 维稠密广播基线仅为：

\[
\rho_{64} \approx 6.2\%.
\]

因此可以说：

```text
Our strongest mainline operates in a genuinely low-bandwidth regime rather than
relying on high-dimensional dense hidden-state communication.
```

### 2. 相比仓库内 MAIC，我们的主线通信更省

在相同 `5m_vs_6m` 环境上，仓库内 MAIC 的执行期通信量约为 `48`，而我们当前攻击流为 `15.9`。

因此可以说：

```text
Compared with the in-repo MAIC implementation, our attack communication branch
uses about one third of the execution-time communication dimensions per
agent-step.
```

### 3. 我们的优势来自“稀疏低维 + 受控注入”，而不是“latent 更小”

这个点很重要。不要把：

```text
MAIC latent_dim = 8
```

误写成：

```text
MAIC communication cost = 8
```

因为在本仓库实现里，真正进入 peer Q-value 的不是 latent 本身，而是 `n_actions` 维动作偏置。因此更准确的说法应当是：

```text
MAIC uses low-dimensional latent modeling internally, but its execution-time
communication carrier is still a dense action-bias tensor over all peers.
```

## 当前表格的边界与注意事项

### 1. 这张表目前是“攻击流主线成本”，不是“全双流总成本”

当前最强主线的低成本叙事主要建立在攻击通信流上，因为：

- 我们这条论文主线的关键机制修复集中在 attack stream；
- 已经有稳定日志支持 `attack_no_comm_prob`、`gate`、`effective_delta` 等指标；
- move stream 目前没有在同一份主线摘要里同步给出完整的末段通信预算统计。

因此，若后续正文要声称“整个双流模型总通信成本也很低”，应当再补一份 move stream 的同口径统计，然后报告：

\[
C_{\mathrm{total}} = C_{\mathrm{attack}} + C_{\mathrm{move}}.
\]

### 2. MAIC 的 48 是执行期动作偏置维数，不包含训练期 MI 建模开销

这张表只比较执行期通信量，不比较：

- 训练期 mutual information 目标；
- latent inference network 的额外参数；
- 额外 forward / backward FLOPs。

如果后面想扩展成“总训练代价对比”，需要另开一张复杂度表。

### 3. 文献方法暂时更适合先放公式框架，再补数字

对于 TarMAC、ATOC、IC3Net、CoDe 等文献方法，当前在本仓库没有对应实现，或者缺少同环境同配置下的 `p_comm / k / d_exec` 观测值。因此更稳妥的做法是：

- 先在 related work 或 experiment setup 里给出它们所属通信族的公式框架；
- 若后续需要做更完整论文表，再按论文原设定补充估算值或公开实现统计。

## 面向文献方法的扩展框架

如果后面要把表扩成“论文里的几类基线方法”，建议按通信家族来组织，而不是强行把所有方法塞成一列数字。

| Family | Typical carrier | Typical cost form |
|---|---|---|
| Dense broadcast (`CommNet`-like) | hidden state / continuous embedding | `C = (N-1) * d_exec` |
| Dense targeted action bias (`MAIC`-like) | per-peer action-space bias | `C = (N-1) * d_action` |
| Sparse targeted attention (`TarMAC` / sparse MAPPO-like) | low-dim message with top-k routing | `C = p_comm * k * d_exec` |
| Hard on-demand group communication (`ATOC` / `IC3Net`-like) | gated low-dim message | `C = p_gate * k * d_exec` |
| Dual-stream selective residual (ours) | attack/move split low-dim residuals | `C = C_attack + C_move` |

这样写的好处是：

- 不会假装我们已经知道所有文献方法在当前实验设置里的精确通信概率；
- 但论文结构上已经有一个统一衡量框架；
- 以后只要补上 `p_comm`、`k` 和 `d_exec`，表就能自然扩展。

## 可直接写进论文的简洁表述

适合写进正文的一段短表述可以是：

```text
We measure communication cost by the expected number of execution-time
communicated dimensions per agent-step,
$C_{\mathrm{exec}} = p_{\mathrm{comm}} \cdot k \cdot d_{\mathrm{exec}}$.
Under this metric, the strongest attack-stream mainline in our paper
(v5b seed2) uses about 15.9 communicated dimensions per agent-step on
SMAC 5m\_vs\_6m, which is roughly one third of the in-repo MAIC
implementation (48.0), and only 6.2\% of a dense 64-dimensional all-to-all
hidden-state broadcast baseline.
```

如果想写得更谨慎一点，可以用：

```text
This comparison should be read as an execution-time communication budget
proxy rather than a full system-complexity measure. In particular, MAIC's
internal latent dimension is not itself the execution communication bandwidth;
the actual communicated carrier in the in-repo implementation is a dense
per-peer action-bias tensor of size $n_{\mathrm{actions}}$.
```

## 下一步建议

如果下一步要把这份框架真正融进论文，我建议直接做两件事：

1. 在实验章节放这张成本表，并把 `v5b seed2` 作为主结果行。
2. 在 related work 或 method 里补一句：本文比较的是“执行期通信预算”，不把训练期 latent inference 复杂度混进同一张表。

这样口径会非常稳，也不容易被 reviewer 抓住“你拿 latent_dim 跟 message dim 混比”的问题。

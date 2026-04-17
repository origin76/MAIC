# V5 Dual-Stream Communication Analysis Report

Date: 2026-04-15

Scope:
- This report summarizes the completed `v5` dual-stream communication experiments exported under `results/sc2/5m_vs_6m`.
- It focuses on backbone quality, warm-start effects, attack-stream and move-stream diagnostics, and which follow-up directions are worth pursuing.
- If there is an additional `distpen` run still in progress but not yet exported to JSON, it is not included here.

## 1. Backbone Context

Before interpreting communication results, the most important control is the quality of the local MAPPO backbone.

### 1.1 Direct officialish training

Config:
- `vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor`

Completed seeds:

| Seed | Peak | Final | Last5 |
| --- | ---: | ---: | ---: |
| 1 | 0.6875 | 0.3125 | 0.5625 |
| 2 | 0.1250 | 0.0625 | 0.0250 |
| 3 | 0.0625 | 0.0000 | 0.0250 |
| Mean | 0.2917 | 0.1250 | 0.2042 |

Interpretation:
- Direct training from scratch can eventually produce a useful policy on a good seed, but overall seed variance is very large.
- The optimization burden is still dominated by learning local micro behavior, survival behavior, and stable PPO dynamics.

### 1.2 Warm-start control

Config:
- `vanilla_mappo_sc2_5m6m_finetune_control_300k_from_relaxactor1404757`

Completed seeds:

| Seed | Peak | Final | Last5 |
| --- | ---: | ---: | ---: |
| 1 | 0.8750 | 0.6875 | 0.6500 |
| 2 | 0.7500 | 0.5625 | 0.6500 |
| 3 | 0.9375 | 0.6250 | 0.6250 |
| Mean | 0.8542 | 0.6250 | 0.6417 |

Interpretation:
- Warm-start from a strong checkpoint changes the entire optimization regime.
- The actor no longer needs to learn basic micro behavior and can spend most updates on refining coordination and stabilizing decision boundaries.
- This is the correct launch point for communication experiments.

## 2. Why Warm-Start Matters for V5

All current `v5` runs are initialized from:
- `results/models/2026-04-13_10-42-26_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor_sc2_5m_vs_6m`
- `init_load_step = 1404757`

This has several practical consequences:

1. Local competence already exists.
The local actor already knows how to kite, stutter-step, retreat, and focus fire at a usable level. Communication is therefore trained as a residual coordination layer rather than as a replacement policy.

2. Gate learning becomes meaningful.
If the backbone is weak, a low gate can mean either "communication is useless" or "the whole policy is still confused". Under warm-start, a low gate is much easier to interpret as a real fusion choice.

3. Credit assignment is cleaner.
Without warm-start, PPO gradients must simultaneously shape local control and communication semantics. That tends to blur whether a failure comes from message content, routing, or the actor itself.

4. Communication is less likely to be scapegoated.
A cold-start communication model often looks bad simply because local policy learning is unfinished. Warm-start makes communication gains and failures far more attributable.

Current conclusion:
- For `v5`, warm-start is not a convenience hack; it is part of the experimental design.
- A direct cold-start `v5` ablation may still be worth running once for rigor, but it should be treated as a control experiment, not as the main training route.

## 3. Completed V5 Results

### 3.1 Summary table

| Variant | Seeds | Mean Peak | Mean Final | Mean Last5 | Main move routing |
| --- | ---: | ---: | ---: | ---: | --- |
| `v5_base` dual-stream targeted fusion | 1 | 0.8750 | 0.8750 | 0.6125 | move top-2, sigmoid gate |
| `v5_top1move` | 1 | 0.7500 | 0.4375 | 0.4875 | move top-1, sigmoid gate |
| `v5_top1move_softplus` | 3 | 0.7708 | 0.6875 | 0.5875 | move top-1, softplus gate |
| `v5_top2move_softplus` | 2 | 0.7812 | 0.5938 | 0.5375 | move top-2, softplus gate |
| `v5_top2move_softplus_movegain` | 1 | 0.8750 | 0.5000 | 0.3875 | move top-2, softplus gate, stronger move fusion |
| `v5_top2move_softplus_distpen` | 1 | 0.6875 | 0.3750 | 0.5000 | move top-2, softplus gate, distance penalty |

### 3.2 Immediate ranking

From a practical perspective:

1. Best single finished run:
- `v5_base`, seed 1
- Peak `0.875`, Final `0.875`, Last5 `0.6125`

2. Best multi-seed stability so far:
- `v5_top1move_softplus`
- Mean Final `0.6875`, Mean Last5 `0.5875`

3. Best new small-sweep direction:
- `v5_top2move_softplus`
- Better than `top1move`, and better than the `movegain` and `distpen` variants as a follow-up path

## 4. Attack Stream vs Move Stream

The core value of `v5` is that it lets us inspect attack coordination and movement coordination separately.

### 4.1 Attack stream

Observed pattern across almost all `v5` runs:

- `targeted_attack_mean_attn_entropy` is always approximately `0.0`
- `targeted_attack_gate_mean` is small but stable, roughly `0.058` to `0.082`
- `targeted_attack_message_norm` is stable, roughly `2.54` to `2.82`

Interpretation:
- The attack stream is already extremely sparse and highly decisive.
- This is consistent with the task structure: "who to attack" is a discrete, sharp coordination variable.
- Attack-side attention is not the current bottleneck.

What changes across runs is not whether the attack stream becomes sparse, but how often the policy chooses to ignore it:
- `targeted_attack_no_comm_prob` varies strongly by seed and variant
- Some runs still use attack communication frequently
- Some runs effectively route to the no-communication token most of the time

This means:
- Attack communication can work
- But the whole policy is not consistently dependent on it
- The system is still willing to fall back to local attack behavior when needed

### 4.2 Move stream

The move stream is the real difficulty.

There are two qualitatively different regimes:

#### Regime A: move top-1

Variants:
- `v5_top1move`
- `v5_top1move_softplus`

Observed properties:
- `targeted_move_mean_attn_entropy ~ 0.0`
- Hard sparse routing really is happening
- `targeted_move_message_norm` is lower and more variable than the top-2 case
- `targeted_move_no_comm_prob` is highly unstable across seeds

Interpretation:
- Top-1 move routing successfully prevents averaging.
- However, it is brittle.
- The model is forced to trust one neighbor, and if that chosen neighbor is not consistently informative, move communication becomes unstable.

Performance evidence:
- `v5_top1move` is clearly worse than `v5_base`
- `v5_top1move_softplus` rescues this partially by making gate behavior smoother, but it does not fundamentally solve the routing brittleness

#### Regime B: move top-2

Variants:
- `v5_base`
- `v5_top2move_softplus`
- `v5_top2move_softplus_movegain`
- `v5_top2move_softplus_distpen`

Observed properties:
- `targeted_move_mean_attn_entropy ~ 0.693`
- This is essentially `ln(2)`, which means the move stream is averaging across two neighbors almost uniformly
- `targeted_move_message_norm` is almost pinned at `2.827+`
- `targeted_move_gate_mean` stays very low, roughly `0.044` to `0.048`

Interpretation:
- The move stream is active, but not selective.
- The network is not collapsing the move branch; instead, it is sending in a stable, fairly high-norm message that is then softly gated.
- The real bottleneck is not "the message is absent", but "the message is too averaged to become physically decisive".

This is the key structural conclusion from `v5` so far:
- Attack coordination prefers hard, nearly deterministic routing
- Movement coordination does not benefit from naively pushing the same hardness
- But leaving movement at a uniform top-2 average is also not ideal

So the open problem is:
- How to make move top-2 selective without turning it into brittle top-1

## 5. Detailed Interpretation of New Soft Variants

### 5.1 `top2move_softplus`

Completed seeds:
- Seed 1: Peak `0.875`, Final `0.625`, Last5 `0.575`
- Seed 2: Peak `0.6875`, Final `0.5625`, Last5 `0.500`

Interpretation:
- This is currently the most reasonable continuation of the `v5` line.
- It preserves the softer move routing of top-2 while avoiding the harshest top-1 brittleness.
- It still has the move averaging problem, but it remains competitive.

### 5.2 `top2move_softplus_movegain`

Completed seed:
- Seed 1: Peak `0.875`, Final `0.500`, Last5 `0.3875`

Interpretation:
- Increasing `move_fusion_scale` from `0.05` to `0.07` does not solve the move bottleneck.
- It produces early high peaks, but worse late-stage stability.
- This strongly suggests that the current problem is not "move communication too weak to be heard".
- The problem is more likely "move communication content and routing remain too averaged".

### 5.3 `top2move_softplus_distpen`

Completed seed:
- Seed 1: Peak `0.6875`, Final `0.375`, Last5 `0.500`

Interpretation:
- A distance penalty coefficient of `0.5` is too aggressive under the current design.
- It biases routing toward local neighbors, but not in a way that improves performance.
- This suggests the move stream cannot be fixed by a heavy local prior alone.

## 6. What Is Worth Doing Next

### 6.1 Worth doing

1. Keep warm-start as the default launch mode.
- This is the cleanest way to study communication as a residual coordination mechanism.

2. Keep the dual-stream factorization.
- The attack stream and move stream clearly behave differently.
- That difference is informative and algorithmically meaningful.

3. Keep softplus gates.
- Compared with the hard top-1 sigmoid version, softplus improves smoothness and usually improves final stability.

4. Continue from `top2move_softplus`, not from `top1move`.
- `top1move` is too brittle.
- `top2move_softplus` is the best current compromise.

5. Try to break move top-2 symmetry without increasing raw fusion strength.
Recommended low-risk directions:
- Add a very small move-attention entropy penalty only on the move stream
- Use a weaker distance prior, such as `0.1` or `0.2` instead of `0.5`
- Add a learnable bias against the no-communication token for the move stream
- Add a very small move gate floor only if diagnostics show chronic under-use

### 6.2 Not worth doing right now

1. Making move routing even harder than top-1.
- Existing evidence already shows that overly hard move routing hurts stability.

2. Simply increasing `move_fusion_scale`.
- The `movegain` run already suggests that "more move influence" is not the right fix.

3. Using a large distance penalty.
- `0.5` does not help and likely over-constrains routing.

4. Switching the main line back to cold-start.
- That would reintroduce confounding between local micro learning and communication learning.

## 7. Recommended Next Experimental Order

If the goal is to keep moving efficiently, the recommended order is:

1. Use `top2move_softplus` as the current `v5` backbone
2. Run more seeds for `top2move_softplus`
3. Add one minimal asymmetry mechanism to move top-2, not a stronger fusion gain
4. Only after that, consider a single cold-start `v5` control run for completeness

The most informative next ablation would be:
- `top2move_softplus + mild move-attention sharpening`

because it directly tests the current hypothesis:
- movement communication is present
- movement communication is not absent
- movement communication is too uniform

## 8. Bottom-Line Conclusions

1. Warm-start is essential for interpretable `v5` experiments.
2. The attack stream is already structurally sound and not the main bottleneck.
3. The move stream is the central challenge.
4. Move top-1 is too brittle.
5. Move top-2 is more promising, but currently averages too evenly across two neighbors.
6. Increasing move fusion strength is not the right next step.
7. The next useful step is not "stronger move communication", but "more selective top-2 move communication".

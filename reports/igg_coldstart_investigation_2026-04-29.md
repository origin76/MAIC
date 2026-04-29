# Cold-Start Communication Investigation: From IGG to Gate Annealing

## Background

MAPPO with dual-stream communication (attack + move) fails when trained from scratch (cold-start) on SMAC 5m_vs_6m, while warmstart from a pre-trained backbone achieves 55-70% win rate with the same architecture. This investigation systematically diagnoses why, and searches for a principled solution that preserves both streams.

**Baseline:** Pure backbone MAPPO (no communication) reaches 46-58% win rate cold-start.

## Experiment Chronology

All experiments use `vanilla_mappo_microcomm_dualstream_counterfactual_usegate` agent, seed=1.

### Phase 1: IGG Mechanism Tuning (G2-G4)

**G2 series** (various runs): Early IGG with counterfactual gain-based gate learning. Multiple variants adjusting soft weighting, temperature, gain normalization. Results: unstable, gates collapse to floor or oscillate.

**G3** (`g3_lightopen_quality`): Reduced sparsity, softer gate initialization. Still fails — gate death spiral: gain≈0 → no opening pressure → gates close → delta gets no gradient → gain stays 0.

**G4** (`g4_fixed`, run 180): Key fixes from G3 — removed soft_weighting, removed gain_norm, bidirectional no_comm loss, attack topk=1, removed weight_decay. Used `gate_init_bias=-2.5` (gate≈0.076), `delta_zero_init=True`.

| Metric | Value |
|---|---|
| Peak test_return | 10.7 |
| Peak test_battle_won | **11.25%** |
| attack_gate | 0.035 (near floor 0.02) |
| move_gate | 0.041 (near floor 0.04) |
| Sustainability | **Never returns to 0 after first win** |
| Train-test gap | test > train (normal CTDE behavior) |

**Key finding:** G4 succeeds by learning to SUPPRESS communication. Gates stay at floor, effective communication is near-zero. The backbone learns local policy with minimal interference. IGG correctly identifies cold-start communication as harmful and keeps gates closed. But the gate never opens — delta_zero_init=True creates a deadlock where delta=0 at start, so there's never positive gain signal to open gates.

### Phase 2: Forcing Communication Open (G5, Plan G, Plan F)

**G5 v1** (run 182): `gate_init_bias=0.0` (starts at 0.5), `delta_zero_init=False`, added L2 normalization on attack delta. Result: test_return ≈9-9.5, test_battle_won ≈2.5-6.25%. Better than completely closed gate but unstable.

**G5 v2**: Fixed attack delta L2 norm (wasn't applied in dualstream agent due to code path issue). Result: actor_grad_norm collapsed (55→0.21), backbone gradient death.

**Plan G** (run 183): `gate_fixed_value=0.5`, no IGG, no sparsity. Test: gate fixed at 0.5, no learning complexity.
- Result: test_battle_won ≈1.25%, effectively 0%. **Proves the architecture itself (attention routing + delta fusion) is the primary noise source**, not gate learning.
- Even with all learning complexity removed, communication noise prevents backbone learning.

**Plan F** (run 184): Policy-gradient gate learning with light sparsity, no IGG.
- Result: test_battle_won ≈1.25%, gates collapse to floor. **Without IGG's gain signal, gates lack direction and collapse.**

### Phase 3: Warmup Strategies (Plan B1-B3)

**B1** (run 185): `gate_fixed_value=0.5`, `comm_warmup_steps=100000` (linear 0→1), no IGG.
- Peak test_battle_won: **15%** (highest of ALL experiments)
- Train and test track together (both rise to 15%, both collapse)
- Collapses to 0% after ~400k steps
- **Warmup allows backbone to reach higher peak capability than G4**, but fixed gate=0.5 eventually destabilizes training

**B2** (run 186): B1 + `comm_detach_backbone=True` + `comm_warmup_exponent=2.0`.
- test_battle_won: **0%** throughout. Never wins a single battle.
- **Detach prevents necessary backbone-comm co-adaptation.** Backbone can't learn to work with communication.

**B3 v1** (run 187): IGG back on, `gate_init_bias=0.0`, `delta_zero_init=False`, `comm_warmup_steps=400000` (exponent=2), IGG losses also scaled by warmup.
- test_battle_won: **0%**. Gate closes too slowly (symmetric warmup suppresses closing pressure).

**B3 v2** (run 188): Asymmetric warmup — IGG gain loss scaled by warmup, sparsity/overpredict NOT scaled. Added overpredict weight=0.05.
- Gate closes quickly to floor (attack=0.020, move=0.056 by ~200k steps)
- Brief peak test_battle_won=10%, then permanent collapse to 0%
- **Severe train-test divergence:** training battle_won=12.15% vs test=0%
- Root cause: `delta_zero_init=False` creates non-zero delta from day 1; during gate closing phase (0.5→0.02), effective communication constantly changes, creating a moving target. Policy learns to depend on stochasticity (multinomial sampling) to handle non-stationarity. Greedy test exposes the dependency.

## Summary of Results

| Experiment | Peak Test Win Rate | Peak Train Win Rate | Sustainability | Train-Test Gap | Gate Behavior |
|---|---|---|---|---|---|
| Pure backbone | 46-58% | — | Stable | — | N/A |
| **G4** | 11.25% | 6.7% | **Never returns to 0** | test > train | Closed, stable |
| G5 v1 | 6.25% | — | — | — | Learned, closing |
| Plan G | 1.25% | — | — | — | Fixed 0.5 |
| Plan F | 1.25% | — | — | — | Collapses |
| **B1** | **15%** | 15% | Collapses to 0 | test ≈ train | Fixed 0.5 |
| B2 | 0% | 0% | — | — | Fixed 0.5 |
| B3 v1 | 0% | 0% | — | — | Closes too slowly |
| B3 v2 | 10% (brief) | 12.15% | Collapses to 0 | **test << train** | Closes fast → floor |

## Key Insights

### 1. Communication noise is the fundamental problem

Plan G proved that even with fixed gate=0.5 and all IGG complexity removed, the attention routing + delta fusion inherently prevents backbone learning during cold-start. The noise is architectural, not learned.

### 2. Warmup works but needs stability

B1's 15% peak (vs G4's 11.25%) proves that warmup allows the backbone to build stronger skills before facing full communication. G4's gate-at-floor approach trades peak capability for stability.

### 3. Gate dynamics create non-stationarity

B3 v2 demonstrated that a gate changing from 0.5→0.02 during training creates a moving target. The policy adapts to the changing environment by relying on stochasticity, producing severe train-test divergence.

### 4. The gate-learning rate isomorphism

Gate value functionally determines the "communication learning rate":
- **Large gate (0.5):** Strong delta gradients → fast coordination learning, but noise accumulation → eventual training collapse
- **Small gate (0.02):** Weak delta gradients → slow learning, but stable backbone training

This mirrors the learning rate decay principle: early training benefits from large steps (strong communication), while late training requires small steps (weak communication) for stable convergence.

### 5. IGG deadlock is structural

IGG can only OPEN gates when it sees positive gain. Positive gain requires communication to be helpful. But during cold-start, communication is inherently harmful (not helpful). So:
- gate_init_bias close to 0 → gate open → communication hurts → gate closes → but closing is slow (symmetric warmup) or creates non-stationarity (asymmetric)
- gate_init_bias strongly negative → gate closed → delta=0 (zero_init) → gain=0 → no signal to open

IGG cannot solve the cold-start problem because it evaluates communication utility based on a backbone that hasn't learned yet. It will always conclude communication is harmful and keep gates at floor.

## Gate Annealing: A Simpler Path

The gate-learning rate isomorphism suggests a natural solution: **anneal the gate like a learning rate.**

```
Phase 1 (0-200k):   gate=0.5, warmup factor 0→1
                    Strong communication → fast coordination learning
                    Backbone builds robust skills with communication support

Phase 2 (200-800k): gate linearly anneals 0.5→0.1
                    Communication contribution gradually decreases
                    Policy transitions from "fast learning" to "stable refinement"

Phase 3 (800k+):    gate=0.1, factor=1.0
                    Weak but non-zero communication for stable fine-tuning
                    Analogous to G4's stable closed-gate regime but with more headroom
```

**Advantages over IGG:**
- No learned gate → no deadlock, no non-stationarity from gate dynamics
- No auxiliary losses (IGG gain, sparsity, overpredict) → no backward noise
- Gate trajectory is deterministic and smooth → backbone experiences a consistent, predictable environment
- Single config parameter (`gate_anneal_start`, `gate_anneal_end`, `gate_anneal_steps`) replaces 10+ IGG hyperparameters

**Why it should work:**
- B1 proved warmup + gate=0.5 can reach 15% (the "fast learning" phase)
- G4 proved stable low-gate regime sustains performance (the "stable refinement" phase)
- Annealing bridges the two: use B1's strength early, transition to G4's stability late
- No IGG means no gate learning instability, no deadlock

**Implementation:** Gate value is computed from t_env using a schedule, similar to warmup_factor. Forward fusion uses `schedule_gate(t_env)` instead of learned gate. All IGG, sparsity, overpredict losses removed. Learner unchanged.

## Experiment Files Reference

| Experiment | Config | Script | Run # |
|---|---|---|---|
| G4 | `...g4_fixed.yaml` | — | 180 |
| G5 v1 | `...g5_delta_random_init_open_gate.yaml` | `.sh` | 182 |
| Plan G | `...g5_plan_g_fixed_gate.yaml` | `.sh` | 183 |
| Plan F | `...g5_plan_f_pg_gate.yaml` | `.sh` | 184 |
| B1 | `...g5_plan_b1_warmup_fixed_gate.yaml` | `.sh` | 185 |
| B2 | `...g5_plan_b2_detach_warmup.yaml` | `.sh` | 186 |
| B3 v1 | `...g5_plan_b3_warmup_igg.yaml` | `.sh` | 187 |
| B3 v2 | `...g5_plan_b3_warmup_igg.yaml` (asymmetric) | `.sh` | 188 |

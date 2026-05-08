# Peer-vs-Local Causal Diagnostic

Date: 2026-05-07

## Context

We diagnosed why attack communication has weak execution-time causal impact in
the `v6_attack_top2_selective_silencebudget_signsplit_correction` checkpoints.
The isolation eval showed `normal ~= no_attack`, so the next question was
whether peer communication is redundant, noisy, or ignored by the action logits.

The diagnostic compares three attack-target distributions at each attack-capable
state:

- `local_top1`: target preferred by the local policy before attack communication.
- `peer_top1`: target preferred by the attention-weighted peer intent.
- `fused_top1`: target preferred after attack communication is fused into attack logits.

The key intervention metric is what happens when `peer_top1 != local_top1`.

Note: the first collected table below used three different model checkpoints
with the diagnostic script's eval seed still fixed to `1`. The script has since
been updated to accept an explicit seed argument. The mechanism-level result is
still meaningful because all three independently trained checkpoints show the
same pattern.

## Raw Diagnostic Results

| Checkpoint | Step | Win | Valid states | Conflict count | Conflict rate | Peer top1 prob | Local top1 prob | No-comm prob | Gate mean | Delta abs | Effective delta abs | Fused follow peer on conflict | Fused stay local on conflict | Attack-only follow peer on conflict | Attack-only stay local on conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed1 signsplit | 250991 | 0.5312 | 9084 | 1060 | 0.1167 | 0.9396 | 0.9129 | 0.0055 | 0.0608 | 0.1030 | 0.0006 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| seed2 signsplit | 276282 | 0.4688 | 10260 | 1168 | 0.1138 | 0.9484 | 0.9497 | 0.0001 | 0.0861 | 0.0765 | 0.0007 | 0.0000 | 1.0000 | 0.0068 | 0.9932 |
| seed3 signsplit | 426683 | 0.6250 | 10444 | 1132 | 0.1084 | 0.8879 | 0.8704 | 0.0001 | 0.0715 | 0.0678 | 0.0005 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |

## Mean Pattern

| Metric | Mean |
| --- | ---: |
| conflict_rate | 0.1130 |
| peer_top1_prob_mean | 0.9253 |
| local_top1_prob_mean | 0.9110 |
| no_comm_prob_mean | 0.0019 |
| gate_mean | 0.0728 |
| delta_abs_mean | 0.0824 |
| effective_delta_abs_mean | 0.0006 |
| fused_follow_peer_on_conflict_rate | 0.0000 |
| fused_stay_local_on_conflict_rate | 1.0000 |
| attack_only_follow_peer_on_conflict_rate | 0.0023 |
| attack_only_stay_local_on_conflict_rate | 0.9977 |

## Interpretation

The diagnosis separates representation from causal action leverage.

1. Peer intent is not redundant. `peer_top1 != local_top1` occurs in about
   11 percent of valid attack states.

2. Peer intent is not obviously weak. The peer top1 confidence is high
   (`~0.93`) and comparable to local top1 confidence (`~0.91`).

3. The stream is not escaping through the no-comm token. The no-comm probability
   is effectively zero after the silence-budget fix.

4. The fused policy ignores peer intent in every observed conflict state:
   `fused_follow_peer_on_conflict = 0` and
   `fused_stay_local_on_conflict = 1`.

5. Removing the learned gate is still insufficient. The `attack_only` path
   follows peer in only about 0.2 percent of conflict states, which means the
   residual direction and scale are too weak, not just the learned gate.

The core mechanism is therefore:

```text
peer intent exists and is confident
    -> residual is injected into attack logits
    -> effective residual is about 6e-4
    -> local attack logits remain dominant
    -> greedy action target never changes
```

## Algorithmic Implication

Further alignment sampling is unlikely to solve the primary bottleneck. The
communication module already produces a peer signal. The missing piece is
communication-to-action leverage.

The next algorithmic variant should directly optimize the attack-logit margin
on peer-local conflict states:

```text
if can_attack and peer_valid and peer_top1 != local_top1 and peer_conf is high:
    fused_logit(peer_top1) >= fused_logit(local_top1) + margin
```

Because `log_softmax(peer) - log_softmax(local)` equals
`logit(peer) - logit(local)`, this can be implemented using gathered attack
log-probabilities without changing the action head interface.

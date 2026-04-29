import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_mappo_microcomm_targeted_fusion_agent import (
    VanillaMAPPOMicroCommTargetedFusionAgent,
)


class VanillaMAPPOMicroCommDualStreamTargetedFusionAgent(
    VanillaMAPPOMicroCommTargetedFusionAgent
):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOMicroCommDualStreamTargetedFusionAgent, self).__init__(
            input_shape, args
        )

        hidden_dim = args.rnn_hidden_dim
        relation_dim = len(self.relation_feature_indices)

        self.move_topk = getattr(args, "move_comm_topk", self.topk)
        self.move_comm_value_dim = getattr(
            args, "move_comm_value_dim", self.comm_value_dim
        )
        self.move_fusion_scale = getattr(args, "move_fusion_scale", 0.05)
        self.move_gate_init_bias = getattr(args, "move_gate_init_bias", -3.0)
        self.move_gate_floor = getattr(args, "move_gate_floor", 0.0)
        self.move_delta_zero_init = getattr(args, "move_delta_zero_init", True)
        self.attack_gate_activation = getattr(args, "attack_gate_activation", "sigmoid")
        self.move_gate_activation = getattr(args, "move_gate_activation", "sigmoid")
        self.attack_gate_softplus_beta = getattr(args, "attack_gate_softplus_beta", 1.0)
        self.move_gate_softplus_beta = getattr(args, "move_gate_softplus_beta", 1.0)
        self.attack_gate_scale = getattr(args, "attack_gate_scale", 1.0)
        self.move_gate_scale = getattr(args, "move_gate_scale", 1.0)
        self.attack_gate_max = getattr(args, "attack_gate_max", None)
        self.move_gate_max = getattr(args, "move_gate_max", None)
        self.gate_fixed_value = getattr(args, "gate_fixed_value", None)
        self.gate_anneal_enabled = bool(getattr(args, "gate_anneal_enabled", False))
        self.gate_anneal_start_value = float(getattr(args, "gate_anneal_start_value", 0.5))
        self.gate_anneal_end_value = float(getattr(args, "gate_anneal_end_value", 0.1))
        self.gate_anneal_start_step = int(getattr(args, "gate_anneal_start_step", 200000))
        self.gate_anneal_steps = int(getattr(args, "gate_anneal_steps", 600000))
        self.move_distance_penalty_coef = getattr(
            args, "move_distance_penalty_coef", 0.0
        )
        self.move_distance_relation_index = getattr(
            args, "move_distance_relation_index", 1
        )
        self.comm_warmup_steps = int(getattr(args, "comm_warmup_steps", 0))
        self.comm_warmup_delay_steps = int(
            getattr(args, "comm_warmup_delay_steps", 0)
        )
        self.comm_warmup_start_factor = float(
            getattr(args, "comm_warmup_start_factor", 1.0)
        )
        self.comm_warmup_end_factor = float(
            getattr(args, "comm_warmup_end_factor", 1.0)
        )
        self.comm_warmup_exponent = float(
            getattr(args, "comm_warmup_exponent", 1.0)
        )
        self.move_readiness_warmup = bool(
            getattr(args, "move_readiness_warmup", False)
        )
        self.move_readiness_factor_floor = float(
            getattr(args, "move_readiness_factor_floor", 1.0)
        )
        self.move_readiness_entropy_low = float(
            getattr(args, "move_readiness_entropy_low", 0.5)
        )
        self.move_readiness_entropy_high = float(
            getattr(args, "move_readiness_entropy_high", 0.5)
        )
        self.move_readiness_no_comm_low = float(
            getattr(args, "move_readiness_no_comm_low", 0.0)
        )
        self.move_readiness_no_comm_high = float(
            getattr(args, "move_readiness_no_comm_high", 0.0)
        )
        self.move_carrier_mode = getattr(args, "move_carrier_mode", "default")
        self.move_enemy_visible_index = getattr(args, "move_enemy_visible_index", 0)
        self.move_enemy_distance_index = getattr(args, "move_enemy_distance_index", 1)
        self.move_ally_visible_index = getattr(args, "move_ally_visible_index", 0)
        self.move_ally_distance_index = getattr(args, "move_ally_distance_index", 1)
        self.move_entropy_target = getattr(args, "move_entropy_target", None)
        self.move_entropy_target_loss_weight = getattr(
            args, "move_entropy_target_loss_weight", 0.0
        )
        self.move_entropy_upper_only = bool(
            getattr(args, "move_entropy_upper_only", False)
        )
        self.attack_attn_temperature = float(
            getattr(args, "attack_attn_temperature", 1.0)
        )
        self.move_attn_temperature = float(
            getattr(args, "move_attn_temperature", 1.0)
        )
        self.attack_attn_score_clip = getattr(args, "attack_attn_score_clip", None)
        self.move_attn_score_clip = getattr(args, "move_attn_score_clip", None)
        self.move_no_comm_target = getattr(args, "move_no_comm_target", None)
        self.move_no_comm_target_loss_weight = float(
            getattr(args, "move_no_comm_target_loss_weight", 0.0)
        )
        # Attack-stream no-comm control. Motivation: with top-k=1, the no-comm
        # token can become an "easy escape hatch" (especially when real scores
        # drift negative), causing the attack stream to abandon communication.
        self.attack_no_comm_score_penalty = float(
            getattr(args, "attack_no_comm_score_penalty", 0.0)
        )
        self.attack_no_comm_target = getattr(args, "attack_no_comm_target", None)
        self.attack_no_comm_target_loss_weight = float(
            getattr(args, "attack_no_comm_target_loss_weight", 0.0)
        )

        # Budget-aware alias: expose "silence budget" as an explicit knob.
        # This is an implementation-level alias (no new mechanism), so we keep
        # existing fields and just reconcile config values here.
        no_comm_budget = getattr(args, "no_comm_budget", None)
        no_comm_budget_loss_weight = getattr(args, "no_comm_budget_loss_weight", None)
        attack_no_comm_budget = getattr(args, "attack_no_comm_budget", None)
        attack_no_comm_budget_loss_weight = getattr(
            args, "attack_no_comm_budget_loss_weight", None
        )
        move_no_comm_budget = getattr(args, "move_no_comm_budget", None)
        move_no_comm_budget_loss_weight = getattr(
            args, "move_no_comm_budget_loss_weight", None
        )

        if (
            attack_no_comm_budget is not None
            and self.attack_no_comm_target is not None
            and float(attack_no_comm_budget) != float(self.attack_no_comm_target)
        ):
            raise ValueError(
                "attack_no_comm_budget ({}) conflicts with attack_no_comm_target ({})".format(
                    attack_no_comm_budget, self.attack_no_comm_target
                )
            )
        if attack_no_comm_budget is None:
            attack_no_comm_budget = self.attack_no_comm_target
        if attack_no_comm_budget is None:
            attack_no_comm_budget = no_comm_budget
        self.attack_no_comm_target = attack_no_comm_budget

        if (
            attack_no_comm_budget_loss_weight is not None
            and self.attack_no_comm_target_loss_weight > 0
            and float(attack_no_comm_budget_loss_weight)
            != float(self.attack_no_comm_target_loss_weight)
        ):
            raise ValueError(
                "attack_no_comm_budget_loss_weight ({}) conflicts with attack_no_comm_target_loss_weight ({})".format(
                    attack_no_comm_budget_loss_weight,
                    self.attack_no_comm_target_loss_weight,
                )
            )
        if attack_no_comm_budget_loss_weight is None:
            attack_no_comm_budget_loss_weight = self.attack_no_comm_target_loss_weight
        if (
            float(attack_no_comm_budget_loss_weight) == 0.0
            and no_comm_budget_loss_weight is not None
        ):
            attack_no_comm_budget_loss_weight = float(no_comm_budget_loss_weight)
        self.attack_no_comm_target_loss_weight = float(
            attack_no_comm_budget_loss_weight
        )

        if (
            move_no_comm_budget is not None
            and self.move_no_comm_target is not None
            and float(move_no_comm_budget) != float(self.move_no_comm_target)
        ):
            raise ValueError(
                "move_no_comm_budget ({}) conflicts with move_no_comm_target ({})".format(
                    move_no_comm_budget, self.move_no_comm_target
                )
            )
        if move_no_comm_budget is None:
            move_no_comm_budget = self.move_no_comm_target
        if move_no_comm_budget is None:
            move_no_comm_budget = no_comm_budget
        self.move_no_comm_target = move_no_comm_budget

        if (
            move_no_comm_budget_loss_weight is not None
            and self.move_no_comm_target_loss_weight > 0
            and float(move_no_comm_budget_loss_weight)
            != float(self.move_no_comm_target_loss_weight)
        ):
            raise ValueError(
                "move_no_comm_budget_loss_weight ({}) conflicts with move_no_comm_target_loss_weight ({})".format(
                    move_no_comm_budget_loss_weight,
                    self.move_no_comm_target_loss_weight,
                )
            )
        if move_no_comm_budget_loss_weight is None:
            move_no_comm_budget_loss_weight = self.move_no_comm_target_loss_weight
        if (
            float(move_no_comm_budget_loss_weight) == 0.0
            and no_comm_budget_loss_weight is not None
        ):
            move_no_comm_budget_loss_weight = float(no_comm_budget_loss_weight)
        self.move_no_comm_target_loss_weight = float(move_no_comm_budget_loss_weight)
        self.move_self_feature_indices = list(
            getattr(args, "move_self_feature_indices", [0])
        )
        self.counterfactual_usegate = bool(
            getattr(args, "counterfactual_usegate", False)
        )
        # Eval-only diagnostic: when enabled, force both fusion gates to 1.0 in
        # `test_mode` to check whether comm features are useful but suppressed.
        self.eval_force_comm_gate_open = bool(
            getattr(args, "eval_force_comm_gate_open", False)
        )
        # Eval-only diagnostic: when enabled, force both fusion gates to 0.0 in
        # `test_mode` (i.e., no comm influence on logits) as a clean ablation.
        self.eval_force_comm_gate_closed = bool(
            getattr(args, "eval_force_comm_gate_closed", False)
        )
        # Eval-only diagnostic: when enabled, disallow selecting the no-comm token
        # in attention (forces communication to always come from real teammates).
        self.eval_disable_no_comm_token = bool(
            getattr(args, "eval_disable_no_comm_token", False)
        )
        # Eval-only diagnostic: disable one stream at a time to isolate causal
        # effect without changing the training graph.
        self.eval_disable_attack_comm = bool(
            getattr(args, "eval_disable_attack_comm", False)
        )
        self.eval_disable_move_comm = bool(
            getattr(args, "eval_disable_move_comm", False)
        )
        if self.eval_force_comm_gate_open and self.eval_force_comm_gate_closed:
            raise ValueError(
                "eval_force_comm_gate_open and eval_force_comm_gate_closed cannot both be True"
            )
        if self.eval_force_comm_gate_open and (
            self.eval_disable_attack_comm or self.eval_disable_move_comm
        ):
            raise ValueError(
                "eval_force_comm_gate_open conflicts with eval_disable_attack_comm/eval_disable_move_comm"
            )

        if len(self.move_self_feature_indices) == 0:
            raise ValueError("move_self_feature_indices must be non-empty")
        if self.semantic_action_offset <= 0:
            raise ValueError("semantic_action_offset must be positive")
        if self.move_entropy_target_loss_weight > 0 and self.move_entropy_target is None:
            raise ValueError(
                "move_entropy_target must be set when move_entropy_target_loss_weight > 0"
            )
        if (
            self.move_no_comm_target_loss_weight > 0
            and self.move_no_comm_target is None
        ):
            raise ValueError(
                "move_no_comm_target must be set when move_no_comm_target_loss_weight > 0"
            )
        if self.move_no_comm_target_loss_weight > 0 and not self.use_no_comm_token:
            raise ValueError(
                "move_no_comm_target_loss_weight requires use_no_comm_token=True"
            )
        if self.attack_no_comm_score_penalty < 0.0:
            raise ValueError("attack_no_comm_score_penalty must be non-negative")
        if (
            self.attack_no_comm_target_loss_weight > 0
            and self.attack_no_comm_target is None
        ):
            raise ValueError(
                "attack_no_comm_target must be set when attack_no_comm_target_loss_weight > 0"
            )
        if self.attack_no_comm_target_loss_weight > 0 and not self.use_no_comm_token:
            raise ValueError(
                "attack_no_comm_target_loss_weight requires use_no_comm_token=True"
            )
        if self.attack_no_comm_score_penalty > 0 and not self.use_no_comm_token:
            raise ValueError(
                "attack_no_comm_score_penalty requires use_no_comm_token=True"
            )
        if self.move_carrier_mode not in {"default", "semantic_threat"}:
            raise ValueError(
                "Unsupported move_carrier_mode '{}'".format(
                    self.move_carrier_mode
                )
            )
        if self.move_readiness_factor_floor <= 0.0 or self.move_readiness_factor_floor > 1.0:
            raise ValueError("move_readiness_factor_floor must be in (0, 1]")
        if self.move_readiness_entropy_high < self.move_readiness_entropy_low:
            raise ValueError("move_readiness_entropy_high must be >= move_readiness_entropy_low")
        if self.move_readiness_no_comm_high < self.move_readiness_no_comm_low:
            raise ValueError("move_readiness_no_comm_high must be >= move_readiness_no_comm_low")
        if self.move_attn_temperature <= 0:
            raise ValueError("move_attn_temperature must be positive")
        if self.attack_attn_temperature <= 0:
            raise ValueError("attack_attn_temperature must be positive")
        if (
            self.attack_attn_score_clip is not None
            and float(self.attack_attn_score_clip) <= 0
        ):
            raise ValueError("attack_attn_score_clip must be positive when set")
        if (
            self.move_attn_score_clip is not None
            and float(self.move_attn_score_clip) <= 0
        ):
            raise ValueError("move_attn_score_clip must be positive when set")

        if self.move_carrier_mode == "semantic_threat":
            self.move_sender_state_dim = len(self.move_self_feature_indices) + 4
        else:
            self.move_sender_state_dim = len(self.move_self_feature_indices) + 1
        self.move_flat_comm_dim = self.attention_heads * self.move_comm_value_dim

        self.move_query_proj = nn.Linear(
            hidden_dim, self.attention_heads * self.head_dim
        )
        self.move_key_pair_proj = nn.Linear(
            hidden_dim + relation_dim, self.attention_heads * self.head_dim
        )
        self.move_value_proj = nn.Linear(
            self.semantic_action_offset + self.move_sender_state_dim,
            self.attention_heads * self.move_comm_value_dim,
        )
        self.move_message_layer_norm = nn.LayerNorm(self.move_flat_comm_dim)

        move_gate_hidden_dim = getattr(args, "move_gate_hidden_dim", hidden_dim)
        move_delta_hidden_dim = getattr(args, "move_delta_hidden_dim", hidden_dim)
        move_fusion_input_dim = hidden_dim + self.move_flat_comm_dim
        self.move_gate = nn.Sequential(
            nn.Linear(move_fusion_input_dim, move_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(move_gate_hidden_dim, 1),
        )
        self.move_delta_head = nn.Sequential(
            nn.Linear(move_fusion_input_dim, move_delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(move_delta_hidden_dim, self.semantic_action_offset),
        )

        if self.use_no_comm_token:
            self.move_null_key = nn.Parameter(
                th.zeros(1, 1, 1, self.attention_heads, self.head_dim)
            )
            self.move_null_value = nn.Parameter(
                th.zeros(1, 1, 1, self.attention_heads, self.move_comm_value_dim)
            )

        nn.init.constant_(self.move_gate[-1].bias, self.move_gate_init_bias)
        if self.move_delta_zero_init:
            nn.init.constant_(self.move_delta_head[-1].weight, 0.0)
            nn.init.constant_(self.move_delta_head[-1].bias, 0.0)

        self._comm_param_prefixes = self._comm_param_prefixes + (
            "move_query_proj",
            "move_key_pair_proj",
            "move_value_proj",
            "move_message_layer_norm",
            "move_gate",
            "move_delta_head",
            "move_null_key",
            "move_null_value",
        )

        self.own_feat_start = None
        self.own_feat_end = None
        self.own_feat_dim = None

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        agent_hidden = h.reshape(bs, self.n_agents, -1)
        step_warmup_factor = self._compute_comm_warmup_factor(kwargs.get("t_env", None))

        raw_obs = kwargs.get("raw_obs", None)
        avail_actions = kwargs.get("avail_actions", None)
        comm_source = agent_hidden.detach() if self.comm_detach_backbone else agent_hidden

        local_logits = self.policy_head(agent_hidden.reshape(bs * self.n_agents, -1)).reshape(
            bs, self.n_agents, self.n_actions
        )
        local_move_logits = local_logits[:, :, : self.semantic_action_offset]
        local_attack_logits = local_logits[:, :, self.semantic_action_offset :]

        move_avail_actions = self._get_move_avail_actions(avail_actions)
        attack_avail_actions = self._get_attack_avail_actions(avail_actions)

        move_probs = self._build_move_probs(local_move_logits, move_avail_actions)
        move_top1_mass = move_probs.max(dim=-1, keepdim=True)[0]
        own_state_features = self._extract_own_state_features(
            raw_obs, bs, agent_hidden.device
        )
        if attack_avail_actions is None:
            can_attack = move_probs.new_ones(bs, self.n_agents, 1)
        else:
            can_attack = (
                attack_avail_actions.sum(dim=-1, keepdim=True) > 0
            ).float()

        move_enemy_pressure = None
        move_ally_support = None
        move_retreat_urgency = None
        move_engage_readiness = None
        if self.move_carrier_mode == "semantic_threat":
            move_enemy_pressure = self._compute_move_enemy_pressure(
                raw_obs, bs, agent_hidden.device
            )
            move_ally_support = self._compute_move_ally_support(
                raw_obs, bs, agent_hidden.device
            )
            own_health_proxy = own_state_features[:, :, :1]
            move_retreat_urgency = move_enemy_pressure * (1.0 - own_health_proxy)
            move_engage_readiness = move_enemy_pressure * can_attack
            move_value_source = th.cat(
                [
                    move_probs,
                    own_state_features,
                    move_retreat_urgency,
                    move_engage_readiness,
                    move_ally_support,
                    move_top1_mass,
                ],
                dim=-1,
            )
        else:
            move_value_source = th.cat(
                [move_probs, own_state_features, move_top1_mass], dim=-1
            )
        if self.intent_detach:
            move_value_source = move_value_source.detach()
        move_sender_values = self.move_value_proj(move_value_source).reshape(
            bs, self.n_agents, self.attention_heads, self.move_comm_value_dim
        )

        attack_probs = self._build_attack_probs(
            local_attack_logits, attack_avail_actions
        )
        attack_top1_mass = attack_probs.max(dim=-1, keepdim=True)[0]
        attack_value_source = th.cat(
            [attack_probs, can_attack, attack_top1_mass], dim=-1
        )
        if self.intent_detach:
            attack_value_source = attack_value_source.detach()
        sender_values = self.value_proj(attack_value_source).reshape(
            bs, self.n_agents, self.attention_heads, self.comm_value_dim
        )

        relation_features = self._extract_relation_features(
            raw_obs, bs, agent_hidden.device
        )
        sender_hidden = comm_source.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        pair_input = th.cat([sender_hidden, relation_features], dim=-1)

        queries = self.query_proj(comm_source).reshape(
            bs, self.n_agents, self.attention_heads, self.head_dim
        )
        pair_keys = self.key_pair_proj(
            pair_input.reshape(bs * self.n_agents * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim)
        attack_scores = (queries.unsqueeze(2) * pair_keys).sum(dim=-1) / (
            self.head_dim ** 0.5
        )
        if self.attack_attn_score_clip is not None:
            clip_value = float(self.attack_attn_score_clip)
            attack_scores = attack_scores.clamp(min=-clip_value, max=clip_value)

        move_queries = self.move_query_proj(comm_source).reshape(
            bs, self.n_agents, self.attention_heads, self.head_dim
        )
        move_pair_keys = self.move_key_pair_proj(
            pair_input.reshape(bs * self.n_agents * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim)
        move_scores = (move_queries.unsqueeze(2) * move_pair_keys).sum(dim=-1) / (
            self.head_dim ** 0.5
        )
        if self.move_attn_score_clip is not None:
            clip_value = float(self.move_attn_score_clip)
            move_scores = move_scores.clamp(min=-clip_value, max=clip_value)

        self_mask = th.eye(
            self.n_agents, device=agent_hidden.device, dtype=th.bool
        ).view(1, self.n_agents, self.n_agents, 1)
        attack_scores = attack_scores.masked_fill(self_mask, -1e10)
        move_scores = move_scores.masked_fill(self_mask, -1e10)

        if (
            self.move_distance_penalty_coef > 0
            and self.move_distance_relation_index < relation_features.size(-1)
        ):
            distance_penalty = relation_features[
                :, :, :, self.move_distance_relation_index
            ].unsqueeze(-1)
            move_scores = move_scores - self.move_distance_penalty_coef * distance_penalty

        disable_null_token = (
            test_mode and self.eval_disable_no_comm_token and self.use_no_comm_token
        )

        if self.use_no_comm_token:
            attack_null_scores = (queries.unsqueeze(2) * self.null_key).sum(dim=-1) / (
                self.head_dim ** 0.5
            )
            # Penalize selecting the no-comm token when the agent can actually
            # attack (i.e., communication about "who to focus" should be used).
            # This is a minimal, purely forward-pass prior that helps prevent
            # late-stage collapse where the attack stream abandons comm by
            # routing to the no-comm token too often.
            if self.attack_no_comm_score_penalty > 0:
                attack_null_scores = (
                    attack_null_scores
                    - self.attack_no_comm_score_penalty * can_attack.unsqueeze(2)
                )
            if self.attack_attn_score_clip is not None:
                clip_value = float(self.attack_attn_score_clip)
                attack_null_scores = attack_null_scores.clamp(
                    min=-clip_value, max=clip_value
                )
            if disable_null_token:
                attack_null_scores = attack_null_scores.new_full(
                    attack_null_scores.shape, -1e10
                )
            attack_scores = th.cat([attack_scores, attack_null_scores], dim=2)

            move_null_scores = (
                move_queries.unsqueeze(2) * self.move_null_key
            ).sum(dim=-1) / (self.head_dim ** 0.5)
            if self.move_attn_score_clip is not None:
                clip_value = float(self.move_attn_score_clip)
                move_null_scores = move_null_scores.clamp(
                    min=-clip_value, max=clip_value
                )
            if disable_null_token:
                move_null_scores = move_null_scores.new_full(
                    move_null_scores.shape, -1e10
                )
            move_scores = th.cat([move_scores, move_null_scores], dim=2)

        attack_scores_masked = self._apply_topk_mask(attack_scores)
        if self.attack_attn_temperature != 1.0:
            attack_scores_masked = attack_scores_masked / self.attack_attn_temperature
        attack_alpha = F.softmax(attack_scores_masked, dim=2)
        move_scores_masked = self._apply_topk_mask_with_k(move_scores, self.move_topk)
        if self.move_attn_temperature != 1.0:
            move_scores_masked = move_scores_masked / self.move_attn_temperature
        move_alpha = F.softmax(move_scores_masked, dim=2)
        move_entropy = self._compute_mean_attention_entropy(move_alpha)
        if self.use_no_comm_token:
            move_no_comm_prob = move_alpha[:, :, -1, :].mean()
        else:
            move_no_comm_prob = move_alpha.new_zeros(())
        (
            move_readiness_factor,
            move_entropy_ready,
            move_no_comm_ready,
        ) = self._compute_move_readiness_factor(move_entropy, move_no_comm_prob)
        move_comm_factor = step_warmup_factor * move_readiness_factor

        expanded_attack_values = sender_values.unsqueeze(1).expand(
            -1, self.n_agents, -1, -1, -1
        )
        expanded_move_values = move_sender_values.unsqueeze(1).expand(
            -1, self.n_agents, -1, -1, -1
        )
        if self.use_no_comm_token:
            attack_null_values = self.null_value.expand(
                bs, self.n_agents, -1, -1, -1
            )
            expanded_attack_values = th.cat(
                [expanded_attack_values, attack_null_values], dim=2
            )

            move_null_values = self.move_null_value.expand(
                bs, self.n_agents, -1, -1, -1
            )
            expanded_move_values = th.cat(
                [expanded_move_values, move_null_values], dim=2
            )

        attack_head_messages = (
            attack_alpha.unsqueeze(-1) * expanded_attack_values
        ).sum(dim=2)
        attack_messages = self.message_layer_norm(
            attack_head_messages.reshape(bs, self.n_agents, -1)
        )

        move_head_messages = (
            move_alpha.unsqueeze(-1) * expanded_move_values
        ).sum(dim=2)
        move_messages = self.move_message_layer_norm(
            move_head_messages.reshape(bs, self.n_agents, -1)
        )

        attack_fusion_input = th.cat([agent_hidden, attack_messages], dim=-1)
        attack_gate_logits = self.attack_gate(
            attack_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, 1)
        raw_attack_gate, attack_gate = self._activate_gate(
            attack_gate_logits,
            activation=self.attack_gate_activation,
            floor=self.attack_gate_floor,
            scale=self.attack_gate_scale,
            softplus_beta=self.attack_gate_softplus_beta,
            max_value=self.attack_gate_max,
        )
        if test_mode:
            if self.eval_force_comm_gate_open:
                attack_gate = th.ones_like(attack_gate)
            elif self.eval_force_comm_gate_closed or self.eval_disable_attack_comm:
                attack_gate = th.zeros_like(attack_gate)
        if self.gate_anneal_enabled:
            scheduled_gate = self._compute_gate_anneal_value(kwargs.get("t_env", None))
            attack_gate = attack_gate.new_full(attack_gate.shape, scheduled_gate)
        elif self.gate_fixed_value is not None:
            attack_gate = attack_gate.new_full(attack_gate.shape, self.gate_fixed_value)
        attack_delta = self.attack_delta_head(
            attack_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.attack_action_dim)
        attack_delta_norm = attack_delta.norm(dim=-1, keepdim=True).detach().clamp(min=1.0)
        attack_delta = attack_delta / attack_delta_norm
        fused_attack_logits = (
            local_attack_logits
            + (self.attack_fusion_scale * step_warmup_factor) * attack_gate * attack_delta
        )
        counterfactual_attack_logits = (
            local_attack_logits
            + (self.attack_fusion_scale * step_warmup_factor) * attack_delta
        )

        move_fusion_input = th.cat([agent_hidden, move_messages], dim=-1)
        move_gate_logits = self.move_gate(
            move_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, 1)
        raw_move_gate, move_gate = self._activate_gate(
            move_gate_logits,
            activation=self.move_gate_activation,
            floor=self.move_gate_floor,
            scale=self.move_gate_scale,
            softplus_beta=self.move_gate_softplus_beta,
            max_value=self.move_gate_max,
        )
        if test_mode:
            if self.eval_force_comm_gate_open:
                move_gate = th.ones_like(move_gate)
            elif self.eval_force_comm_gate_closed or self.eval_disable_move_comm:
                move_gate = th.zeros_like(move_gate)
        if self.gate_anneal_enabled:
            scheduled_gate = self._compute_gate_anneal_value(kwargs.get("t_env", None))
            move_gate = move_gate.new_full(move_gate.shape, scheduled_gate)
        elif self.gate_fixed_value is not None:
            move_gate = move_gate.new_full(move_gate.shape, self.gate_fixed_value)
        move_delta = self.move_delta_head(
            move_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.semantic_action_offset)
        move_delta_norm = move_delta.norm(dim=-1, keepdim=True).detach().clamp(min=1.0)
        move_delta = move_delta / move_delta_norm
        fused_move_logits = (
            local_move_logits
            + (self.move_fusion_scale * move_comm_factor) * move_gate * move_delta
        )
        counterfactual_move_logits = (
            local_move_logits
            + (self.move_fusion_scale * move_comm_factor) * move_delta
        )

        final_logits = th.cat([fused_move_logits, fused_attack_logits], dim=-1)

        returns = {}
        if kwargs.get("train_mode", False):
            attack_entropy = self._compute_mean_attention_entropy(attack_alpha)
            if self.attention_entropy_loss_weight > 0:
                returns["attention_entropy_loss"] = (
                    0.5 * (attack_entropy + move_entropy)
                ) * self.attention_entropy_loss_weight
            attack_no_comm_gap = None
            if self.attack_no_comm_target_loss_weight > 0:
                if self.use_no_comm_token:
                    # Only penalize no-comm when the agent can attack (otherwise,
                    # no-comm is expected and shouldn't be discouraged).
                    attack_no_comm_prob = attack_alpha[:, :, -1, :].mean(dim=-1)
                    attack_can = can_attack.squeeze(-1)
                    denom = attack_can.sum().clamp(min=1.0)
                    attack_no_comm_prob_active = (
                        (attack_no_comm_prob * attack_can).sum() / denom
                    )
                else:
                    attack_no_comm_prob_active = attack_alpha.new_zeros(())
                target_no_comm = attack_no_comm_prob_active.new_tensor(
                    float(self.attack_no_comm_target)
                )
                attack_no_comm_gap = attack_no_comm_prob_active - target_no_comm
                returns["attack_no_comm_loss"] = (
                    (self.attack_no_comm_target_loss_weight * step_warmup_factor)
                    * attack_no_comm_gap.clamp(min=0.0).pow(2)
                )
            move_entropy_gap = None
            if self.move_entropy_target_loss_weight > 0:
                target_entropy = move_entropy.new_tensor(float(self.move_entropy_target))
                move_entropy_gap = move_entropy - target_entropy
                move_entropy_penalty = (
                    move_entropy_gap.clamp(min=0.0)
                    if self.move_entropy_upper_only
                    else move_entropy_gap
                )
                returns["move_selective_entropy_loss"] = (
                    (self.move_entropy_target_loss_weight * move_comm_factor)
                    * move_entropy_penalty.pow(2)
                )
            move_no_comm_gap = None
            if self.move_no_comm_target_loss_weight > 0:
                target_no_comm = move_no_comm_prob.new_tensor(
                    float(self.move_no_comm_target)
                )
                move_no_comm_gap = move_no_comm_prob - target_no_comm
                # Penalise in both directions: too little OR too much silence.
                # Prior code only clamped min=0 (punish when gap > 0, i.e.
                # no_comm > target).  When the stream under-uses silence
                # (gap < 0) the squared penalty still applies.
                returns["move_no_comm_loss"] = (
                    (self.move_no_comm_target_loss_weight * move_comm_factor)
                    * move_no_comm_gap.pow(2)
                )
            if (
                self.counterfactual_usegate
                and kwargs.get("collect_sequence_data", False)
                and step_warmup_factor > 0
            ):
                chosen_actions = kwargs.get("chosen_actions", None)
                with th.no_grad():
                    attack_counterfactual_logits = th.cat(
                        [local_move_logits, counterfactual_attack_logits], dim=-1
                    )
                    move_counterfactual_logits = th.cat(
                        [counterfactual_move_logits, local_attack_logits], dim=-1
                    )
                    local_action_top1, _ = self._masked_argmax(
                        local_logits, avail_actions
                    )
                    fused_action_top1, _ = self._masked_argmax(
                        final_logits, avail_actions
                    )
                    attack_only_action_top1, _ = self._masked_argmax(
                        attack_counterfactual_logits, avail_actions
                    )
                    move_only_action_top1, _ = self._masked_argmax(
                        move_counterfactual_logits, avail_actions
                    )
                    local_attack_top1, attack_can_mask = self._masked_argmax(
                        local_attack_logits, attack_avail_actions
                    )
                    fused_attack_top1, _ = self._masked_argmax(
                        fused_attack_logits, attack_avail_actions
                    )
                    attack_only_top1, _ = self._masked_argmax(
                        counterfactual_attack_logits, attack_avail_actions
                    )
                    (
                        local_attack_agreement,
                        attack_pair_valid,
                    ) = self._compute_pairwise_agreement(
                        local_attack_top1, attack_can_mask
                    )
                    fused_attack_agreement, _ = self._compute_pairwise_agreement(
                        fused_attack_top1, attack_can_mask
                    )
                    attack_only_agreement, _ = self._compute_pairwise_agreement(
                        attack_only_top1, attack_can_mask
                    )
                    if chosen_actions is not None:
                        returns["seq_counterfactual_local_logp"] = self._compute_chosen_log_probs(
                            local_logits, avail_actions, chosen_actions
                        )
                        returns["seq_counterfactual_fused_logp"] = self._compute_chosen_log_probs(
                            final_logits, avail_actions, chosen_actions
                        )
                        returns["seq_counterfactual_attack_logp"] = self._compute_chosen_log_probs(
                            attack_counterfactual_logits, avail_actions, chosen_actions
                        )
                        returns["seq_counterfactual_move_logp"] = self._compute_chosen_log_probs(
                            move_counterfactual_logits, avail_actions, chosen_actions
                        )
                    else:
                        # Backward-compatible fallback for callers that don't pass chosen actions.
                        returns[
                            "seq_counterfactual_local_policy"
                        ] = self._build_full_policy_probs(local_logits, avail_actions)
                        returns[
                            "seq_counterfactual_attack_policy"
                        ] = self._build_full_policy_probs(
                            attack_counterfactual_logits, avail_actions
                        )
                        returns[
                            "seq_counterfactual_move_policy"
                        ] = self._build_full_policy_probs(
                            move_counterfactual_logits, avail_actions
                        )
                returns["seq_counterfactual_action_flip_fused"] = (
                    local_action_top1 != fused_action_top1
                ).float()
                returns["seq_counterfactual_action_flip_attack_only"] = (
                    local_action_top1 != attack_only_action_top1
                ).float()
                returns["seq_counterfactual_action_flip_move_only"] = (
                    local_action_top1 != move_only_action_top1
                ).float()
                returns["seq_counterfactual_attack_can_mask"] = attack_can_mask
                returns["seq_counterfactual_attack_pair_valid"] = attack_pair_valid
                returns["seq_counterfactual_attack_target_flip"] = (
                    (local_attack_top1 != fused_attack_top1).float() * attack_can_mask
                )
                returns["seq_counterfactual_attack_target_flip_attack_only"] = (
                    (local_attack_top1 != attack_only_top1).float()
                    * attack_can_mask
                )
                returns["seq_counterfactual_attack_target_agreement_local"] = (
                    local_attack_agreement
                )
                returns["seq_counterfactual_attack_target_agreement_fused"] = (
                    fused_attack_agreement
                )
                returns["seq_counterfactual_attack_target_agreement_attack_only"] = (
                    attack_only_agreement
                )
                returns[
                    "seq_counterfactual_attack_usegate_pred"
                ] = th.sigmoid(attack_gate_logits)
                returns[
                    "seq_counterfactual_move_usegate_pred"
                ] = th.sigmoid(move_gate_logits)
            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(
                    attack_alpha=attack_alpha,
                    move_alpha=move_alpha,
                    relation_features=relation_features,
                    attack_messages=attack_messages,
                    move_messages=move_messages,
                    attack_gate=attack_gate,
                    raw_attack_gate=raw_attack_gate,
                    attack_delta=attack_delta,
                    attack_probs=attack_probs,
                    move_gate=move_gate,
                    raw_move_gate=raw_move_gate,
                    move_delta=move_delta,
                    move_probs=move_probs,
                    own_state_features=own_state_features,
                    move_entropy_gap=move_entropy_gap,
                    move_no_comm_gap=move_no_comm_gap,
                    move_enemy_pressure=move_enemy_pressure,
                    move_ally_support=move_ally_support,
                    move_retreat_urgency=move_retreat_urgency,
                    move_engage_readiness=move_engage_readiness,
                    attack_no_comm_gap=attack_no_comm_gap,
                    step_warmup_factor=step_warmup_factor,
                    move_readiness_factor=move_readiness_factor,
                    move_entropy_ready=move_entropy_ready,
                    move_no_comm_ready=move_no_comm_ready,
                    move_comm_factor=move_comm_factor,
                )

        return final_logits.reshape(bs * self.n_agents, self.n_actions), h, returns

    def _get_move_avail_actions(self, avail_actions):
        if avail_actions is None:
            return None
        return avail_actions[:, :, : self.semantic_action_offset]

    def _build_move_probs(self, move_logits, move_avail_actions):
        if move_avail_actions is None or not self.intent_mask_unavailable:
            return F.softmax(move_logits, dim=-1)

        masked_logits = move_logits.masked_fill(move_avail_actions == 0, -1e10)
        move_probs = F.softmax(masked_logits, dim=-1)
        valid_mask = (move_avail_actions.sum(dim=-1, keepdim=True) > 0).float()
        return move_probs * valid_mask

    def _build_full_policy_probs(self, logits, avail_actions):
        if avail_actions is None:
            return F.softmax(logits, dim=-1)

        masked_logits = logits.masked_fill(avail_actions == 0, -1e10)
        policy = F.softmax(masked_logits, dim=-1)
        valid_mask = (avail_actions.sum(dim=-1, keepdim=True) > 0).float()
        return policy * valid_mask

    def _compute_chosen_log_probs(self, logits, avail_actions, chosen_actions):
        if chosen_actions.dim() == logits.dim() - 1:
            chosen_actions = chosen_actions.unsqueeze(-1)

        if avail_actions is None:
            log_probs = F.log_softmax(logits, dim=-1)
            return th.gather(log_probs, dim=-1, index=chosen_actions).squeeze(-1)

        masked_logits = logits.masked_fill(avail_actions == 0, -1e10)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        chosen_log_probs = th.gather(log_probs, dim=-1, index=chosen_actions).squeeze(-1)
        valid_mask = (avail_actions.sum(dim=-1) > 0).float()
        return chosen_log_probs * valid_mask

    def _masked_argmax(self, logits, avail_actions):
        if avail_actions is None:
            top_idx = logits.argmax(dim=-1)
            valid_mask = top_idx.new_ones(top_idx.shape, dtype=logits.dtype)
            return top_idx, valid_mask

        masked_logits = logits.masked_fill(avail_actions == 0, -1e10)
        top_idx = masked_logits.argmax(dim=-1)
        valid_mask = (avail_actions.sum(dim=-1) > 0).float()
        return top_idx, valid_mask

    def _compute_pairwise_agreement(self, top_idx, valid_mask):
        pair_valid = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)
        eye = th.eye(
            top_idx.size(-1), device=top_idx.device, dtype=pair_valid.dtype
        ).unsqueeze(0)
        pair_valid = pair_valid * (1.0 - eye)
        same_target = (
            (top_idx.unsqueeze(-1) == top_idx.unsqueeze(-2)).float() * pair_valid
        )
        pair_denom = pair_valid.sum(dim=(-1, -2))
        agreement = th.where(
            pair_denom > 0,
            same_target.sum(dim=(-1, -2)) / pair_denom.clamp(min=1.0),
            pair_denom.new_zeros(pair_denom.shape),
        )
        pair_valid_flag = (pair_denom > 0).float()
        return agreement, pair_valid_flag

    def _extract_own_state_features(self, raw_obs, batch_size, device):
        own_feat_count = len(self.move_self_feature_indices)
        if raw_obs is None:
            return th.zeros(
                batch_size, self.n_agents, own_feat_count, device=device
            )

        self._maybe_init_obs_layout(raw_obs.size(-1))
        self._maybe_init_own_layout(raw_obs.size(-1))

        own_feats = raw_obs[:, :, self.own_feat_start : self.own_feat_end]
        return own_feats[:, :, self.move_self_feature_indices]

    def _extract_enemy_features(self, raw_obs, batch_size, device):
        if raw_obs is None:
            return th.zeros(
                batch_size,
                self.n_agents,
                self.attack_action_dim,
                2,
                device=device,
            )

        self._maybe_init_obs_layout(raw_obs.size(-1))
        enemy_flat = raw_obs[:, :, self.enemy_feat_start : self.enemy_feat_end]
        return enemy_flat.reshape(
            raw_obs.size(0), self.n_agents, self.attack_action_dim, self.enemy_feat_dim
        )

    def _extract_sender_ally_features(self, raw_obs, batch_size, device):
        if raw_obs is None:
            return th.zeros(
                batch_size,
                self.n_agents,
                self.n_agents - 1,
                2,
                device=device,
            )

        self._maybe_init_obs_layout(raw_obs.size(-1))
        ally_flat = raw_obs[:, :, self.ally_feat_start : self.ally_feat_end]
        return ally_flat.reshape(
            raw_obs.size(0), self.n_agents, self.n_agents - 1, self.ally_feat_dim
        )

    def _compute_move_enemy_pressure(self, raw_obs, batch_size, device):
        enemy_feats = self._extract_enemy_features(raw_obs, batch_size, device)
        visible = enemy_feats[:, :, :, self.move_enemy_visible_index]
        distance = enemy_feats[:, :, :, self.move_enemy_distance_index]
        closeness = visible * th.clamp(1.0 - distance, min=0.0)
        return closeness.max(dim=-1, keepdim=True)[0]

    def _compute_move_ally_support(self, raw_obs, batch_size, device):
        ally_feats = self._extract_sender_ally_features(raw_obs, batch_size, device)
        visible = ally_feats[:, :, :, self.move_ally_visible_index]
        distance = ally_feats[:, :, :, self.move_ally_distance_index]
        closeness = visible * th.clamp(1.0 - distance, min=0.0)
        return closeness.mean(dim=-1, keepdim=True)

    def _compute_comm_warmup_factor(self, t_env):
        if self.comm_warmup_steps <= 0:
            return 1.0

        delay_steps = max(0, self.comm_warmup_delay_steps)
        if t_env is None:
            progress = 1.0
        else:
            shifted_t = float(t_env) - float(delay_steps)
            progress = float(
                max(0.0, min(1.0, shifted_t / float(self.comm_warmup_steps)))
            )

        nonlinear_progress = progress ** self.comm_warmup_exponent
        return (
            self.comm_warmup_start_factor
            + (self.comm_warmup_end_factor - self.comm_warmup_start_factor) * nonlinear_progress
        )

    def _compute_gate_anneal_value(self, t_env):
        if not self.gate_anneal_enabled or t_env is None:
            return self.gate_anneal_start_value
        start_step = self.gate_anneal_start_step
        if t_env < start_step:
            return self.gate_anneal_start_value
        progress = min(1.0, (t_env - start_step) / max(1, self.gate_anneal_steps))
        return (
            self.gate_anneal_start_value
            + (self.gate_anneal_end_value - self.gate_anneal_start_value) * progress
        )

    def _compute_move_readiness_factor(self, move_entropy, move_no_comm_prob):
        if not self.move_readiness_warmup:
            one = move_entropy.new_tensor(1.0)
            return one, one, one

        entropy_ready = self._compute_descending_readiness(
            move_entropy,
            self.move_readiness_entropy_low,
            self.move_readiness_entropy_high,
        )
        if self.use_no_comm_token:
            no_comm_ready = self._compute_descending_readiness(
                move_no_comm_prob,
                self.move_readiness_no_comm_low,
                self.move_readiness_no_comm_high,
            )
        else:
            no_comm_ready = move_entropy.new_tensor(1.0)

        readiness_core = th.minimum(entropy_ready, no_comm_ready)
        readiness_factor = (
            self.move_readiness_factor_floor
            + (1.0 - self.move_readiness_factor_floor) * readiness_core
        )
        return readiness_factor, entropy_ready, no_comm_ready

    def _compute_descending_readiness(self, value, low, high):
        if high <= low:
            return (value <= low).float()

        high_tensor = value.new_tensor(float(high))
        low_tensor = value.new_tensor(float(low))
        readiness = (high_tensor - value) / (high_tensor - low_tensor)
        return readiness.clamp(min=0.0, max=1.0)

    def _maybe_init_own_layout(self, obs_shape):
        if self.own_feat_dim is not None:
            return

        if not self._obs_layout_ready:
            self._maybe_init_obs_layout(obs_shape)

        self.own_feat_start = self.ally_feat_end
        self.own_feat_end = obs_shape
        self.own_feat_dim = self.own_feat_end - self.own_feat_start

        if self.own_feat_dim <= 0:
            raise ValueError("Unable to infer own feature slice from obs dim")
        if max(self.move_self_feature_indices) >= self.own_feat_dim:
            raise ValueError(
                "move_self_feature_indices exceed inferred own feature dim {}".format(
                    self.own_feat_dim
                )
            )

    def _apply_topk_mask_with_k(self, scores, topk):
        max_options = scores.size(2)
        k = min(max(1, topk), max_options)
        if k >= max_options:
            return scores

        scores_perm = scores.permute(0, 1, 3, 2)
        topk_indices = scores_perm.topk(k=k, dim=-1).indices
        keep_mask = th.zeros_like(scores_perm, dtype=th.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        scores_perm = scores_perm.masked_fill(~keep_mask, -1e10)
        return scores_perm.permute(0, 1, 3, 2)

    def _activate_gate(
        self,
        gate_logits,
        activation,
        floor,
        scale,
        softplus_beta,
        max_value,
    ):
        if activation == "sigmoid":
            raw_gate = th.sigmoid(gate_logits)
            gate = floor + (1.0 - floor) * raw_gate
        elif activation == "softplus":
            raw_gate = F.softplus(gate_logits, beta=softplus_beta)
            gate = floor + scale * raw_gate
        else:
            raise ValueError("Unsupported gate activation '{}'".format(activation))

        if max_value is not None:
            gate = gate.clamp(max=max_value)

        return raw_gate, gate

    def build_logging_payload(
        self,
        attack_alpha,
        move_alpha,
        relation_features,
        attack_messages,
        move_messages,
        attack_gate,
        raw_attack_gate,
        attack_delta,
        attack_probs,
        move_gate,
        raw_move_gate,
        move_delta,
        move_probs,
        own_state_features,
        move_entropy_gap=None,
        move_no_comm_gap=None,
        attack_no_comm_gap=None,
        move_enemy_pressure=None,
        move_ally_support=None,
        move_retreat_urgency=None,
        move_engage_readiness=None,
        step_warmup_factor=1.0,
        move_readiness_factor=1.0,
        move_entropy_ready=1.0,
        move_no_comm_ready=1.0,
        move_comm_factor=1.0,
    ):
        logs = {}

        detached_attack_alpha = attack_alpha.detach()
        detached_move_alpha = move_alpha.detach()

        attack_head_entropy = -(
            th.clamp(detached_attack_alpha, min=1e-8)
            * th.log(th.clamp(detached_attack_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))
        move_head_entropy = -(
            th.clamp(detached_move_alpha, min=1e-8)
            * th.log(th.clamp(detached_move_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))

        logs["Scalar_targeted_mean_attn_entropy"] = 0.5 * (
            attack_head_entropy.mean() + move_head_entropy.mean()
        )
        logs["Scalar_targeted_attack_mean_attn_entropy"] = attack_head_entropy.mean()
        logs["Scalar_targeted_move_mean_attn_entropy"] = move_head_entropy.mean()

        logs["Scalar_targeted_attack_gate_mean"] = attack_gate.detach().mean()
        logs["Scalar_targeted_attack_gate_raw_mean"] = raw_attack_gate.detach().mean()
        logs["Scalar_targeted_attack_delta_norm"] = attack_delta.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_attack_delta_abs_mean"] = attack_delta.detach().abs().mean()
        logs["Scalar_targeted_attack_message_norm"] = attack_messages.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_attack_no_comm_prob"] = (
            detached_attack_alpha[:, :, -1, :].mean()
            if self.use_no_comm_token
            else th.tensor(0.0, device=attack_alpha.device)
        )
        if self.attack_no_comm_target is not None:
            logs["Scalar_targeted_attack_no_comm_target"] = th.tensor(
                float(self.attack_no_comm_target), device=attack_alpha.device
            )
            logs["Scalar_targeted_attack_no_comm_budget"] = th.tensor(
                float(self.attack_no_comm_target), device=attack_alpha.device
            )
        logs["Scalar_targeted_attack_no_comm_target_loss_weight"] = th.tensor(
            float(self.attack_no_comm_target_loss_weight), device=attack_alpha.device
        )
        logs["Scalar_targeted_attack_no_comm_budget_loss_weight"] = th.tensor(
            float(self.attack_no_comm_target_loss_weight), device=attack_alpha.device
        )
        logs["Scalar_targeted_attack_no_comm_score_penalty"] = th.tensor(
            float(self.attack_no_comm_score_penalty), device=attack_alpha.device
        )
        if attack_no_comm_gap is not None:
            logs["Scalar_targeted_attack_no_comm_gap"] = attack_no_comm_gap.detach()
        logs["Scalar_targeted_attack_intent_top1_mass"] = attack_probs.detach().max(dim=-1)[0].mean()
        logs["Scalar_targeted_attack_edge_budget_ratio"] = th.tensor(
            float(min(max(1, self.topk), attack_alpha.size(2))) / float(attack_alpha.size(2)),
            device=attack_alpha.device,
        )

        logs["Scalar_targeted_move_gate_mean"] = move_gate.detach().mean()
        logs["Scalar_targeted_move_gate_raw_mean"] = raw_move_gate.detach().mean()
        logs["Scalar_targeted_move_delta_norm"] = move_delta.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_move_delta_abs_mean"] = move_delta.detach().abs().mean()
        logs["Scalar_targeted_move_message_norm"] = move_messages.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_move_no_comm_prob"] = (
            detached_move_alpha[:, :, -1, :].mean()
            if self.use_no_comm_token
            else th.tensor(0.0, device=move_alpha.device)
        )
        logs["Scalar_targeted_move_intent_top1_mass"] = move_probs.detach().max(dim=-1)[0].mean()
        logs["Scalar_targeted_move_edge_budget_ratio"] = th.tensor(
            float(min(max(1, self.move_topk), move_alpha.size(2))) / float(move_alpha.size(2)),
            device=move_alpha.device,
        )
        logs["Scalar_targeted_move_own_state_mean"] = own_state_features.detach().mean()
        if self.move_entropy_target is not None:
            logs["Scalar_targeted_move_entropy_target"] = th.tensor(
                float(self.move_entropy_target), device=move_alpha.device
            )
        logs["Scalar_targeted_move_entropy_target_loss_weight"] = th.tensor(
            float(self.move_entropy_target_loss_weight), device=move_alpha.device
        )
        logs["Scalar_targeted_move_entropy_upper_only"] = th.tensor(
            1.0 if self.move_entropy_upper_only else 0.0, device=move_alpha.device
        )
        logs["Scalar_targeted_move_attn_temperature"] = th.tensor(
            float(self.move_attn_temperature), device=move_alpha.device
        )
        logs["Scalar_targeted_attack_attn_temperature"] = th.tensor(
            float(self.attack_attn_temperature), device=attack_alpha.device
        )
        if self.attack_attn_score_clip is not None:
            logs["Scalar_targeted_attack_attn_score_clip"] = th.tensor(
                float(self.attack_attn_score_clip), device=attack_alpha.device
            )
        if self.move_attn_score_clip is not None:
            logs["Scalar_targeted_move_attn_score_clip"] = th.tensor(
                float(self.move_attn_score_clip), device=move_alpha.device
            )
        if move_entropy_gap is not None:
            logs["Scalar_targeted_move_entropy_gap"] = move_entropy_gap.detach()
        if self.move_no_comm_target is not None:
            logs["Scalar_targeted_move_no_comm_target"] = th.tensor(
                float(self.move_no_comm_target), device=move_alpha.device
            )
            logs["Scalar_targeted_move_no_comm_budget"] = th.tensor(
                float(self.move_no_comm_target), device=move_alpha.device
            )
        logs["Scalar_targeted_move_no_comm_target_loss_weight"] = th.tensor(
            float(self.move_no_comm_target_loss_weight), device=move_alpha.device
        )
        logs["Scalar_targeted_move_no_comm_budget_loss_weight"] = th.tensor(
            float(self.move_no_comm_target_loss_weight), device=move_alpha.device
        )
        if move_no_comm_gap is not None:
            logs["Scalar_targeted_move_no_comm_gap"] = move_no_comm_gap.detach()
        if move_enemy_pressure is not None:
            logs["Scalar_targeted_move_enemy_pressure"] = move_enemy_pressure.detach().mean()
        if move_ally_support is not None:
            logs["Scalar_targeted_move_ally_support"] = move_ally_support.detach().mean()
        if move_retreat_urgency is not None:
            logs["Scalar_targeted_move_retreat_urgency"] = move_retreat_urgency.detach().mean()
        if move_engage_readiness is not None:
            logs["Scalar_targeted_move_engage_readiness"] = move_engage_readiness.detach().mean()
        logs["Scalar_targeted_comm_warmup_factor"] = th.tensor(
            float(step_warmup_factor), device=move_alpha.device
        )
        logs["Scalar_targeted_move_readiness_factor"] = (
            move_readiness_factor.detach()
            if isinstance(move_readiness_factor, th.Tensor)
            else th.tensor(float(move_readiness_factor), device=move_alpha.device)
        )
        logs["Scalar_targeted_move_entropy_ready"] = (
            move_entropy_ready.detach()
            if isinstance(move_entropy_ready, th.Tensor)
            else th.tensor(float(move_entropy_ready), device=move_alpha.device)
        )
        logs["Scalar_targeted_move_no_comm_ready"] = (
            move_no_comm_ready.detach()
            if isinstance(move_no_comm_ready, th.Tensor)
            else th.tensor(float(move_no_comm_ready), device=move_alpha.device)
        )
        logs["Scalar_targeted_move_comm_factor"] = (
            move_comm_factor.detach()
            if isinstance(move_comm_factor, th.Tensor)
            else th.tensor(float(move_comm_factor), device=move_alpha.device)
        )

        logs["Scalar_targeted_relation_visible_ratio"] = relation_features[:, :, :, 0].detach().mean()
        logs["Scalar_targeted_no_comm_prob"] = 0.5 * (
            logs["Scalar_targeted_attack_no_comm_prob"]
            + logs["Scalar_targeted_move_no_comm_prob"]
        )
        logs["Scalar_targeted_message_norm"] = 0.5 * (
            logs["Scalar_targeted_attack_message_norm"]
            + logs["Scalar_targeted_move_message_norm"]
        )
        logs["Scalar_targeted_edge_budget_ratio"] = 0.5 * (
            logs["Scalar_targeted_attack_edge_budget_ratio"]
            + logs["Scalar_targeted_move_edge_budget_ratio"]
        )
        logs["Scalar_message_norm"] = logs["Scalar_targeted_message_norm"]

        for head_idx in range(self.attention_heads):
            logs["Scalar_targeted_attack_head_{}_entropy".format(head_idx)] = attack_head_entropy[head_idx]
            logs["Scalar_targeted_move_head_{}_entropy".format(head_idx)] = move_head_entropy[head_idx]
            if self.log_attention_maps:
                logs["Histogram_targeted_attack_head_{}_attention".format(head_idx)] = detached_attack_alpha[:, :, :, head_idx]
                logs["Histogram_targeted_move_head_{}_attention".format(head_idx)] = detached_move_alpha[:, :, :, head_idx]

        return logs

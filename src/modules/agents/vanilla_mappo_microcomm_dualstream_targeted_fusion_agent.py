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
        self.move_no_comm_target = getattr(args, "move_no_comm_target", None)
        self.move_no_comm_target_loss_weight = getattr(
            args, "move_no_comm_target_loss_weight", 0.0
        )
        self.move_self_feature_indices = list(
            getattr(args, "move_self_feature_indices", [0])
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

        move_queries = self.move_query_proj(comm_source).reshape(
            bs, self.n_agents, self.attention_heads, self.head_dim
        )
        move_pair_keys = self.move_key_pair_proj(
            pair_input.reshape(bs * self.n_agents * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim)
        move_scores = (move_queries.unsqueeze(2) * move_pair_keys).sum(dim=-1) / (
            self.head_dim ** 0.5
        )

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

        if self.use_no_comm_token:
            attack_null_scores = (queries.unsqueeze(2) * self.null_key).sum(dim=-1) / (
                self.head_dim ** 0.5
            )
            attack_scores = th.cat([attack_scores, attack_null_scores], dim=2)

            move_null_scores = (
                move_queries.unsqueeze(2) * self.move_null_key
            ).sum(dim=-1) / (self.head_dim ** 0.5)
            move_scores = th.cat([move_scores, move_null_scores], dim=2)

        attack_alpha = F.softmax(self._apply_topk_mask(attack_scores), dim=2)
        move_alpha = F.softmax(
            self._apply_topk_mask_with_k(move_scores, self.move_topk), dim=2
        )
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
        attack_delta = self.attack_delta_head(
            attack_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.attack_action_dim)
        fused_attack_logits = (
            local_attack_logits
            + (self.attack_fusion_scale * step_warmup_factor) * attack_gate * attack_delta
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
        move_delta = self.move_delta_head(
            move_fusion_input.reshape(bs * self.n_agents, -1)
        ).reshape(bs, self.n_agents, self.semantic_action_offset)
        fused_move_logits = (
            local_move_logits
            + (self.move_fusion_scale * move_comm_factor) * move_gate * move_delta
        )

        final_logits = th.cat([fused_move_logits, fused_attack_logits], dim=-1)

        returns = {}
        if kwargs.get("train_mode", False):
            attack_entropy = self._compute_mean_attention_entropy(attack_alpha)
            if self.attention_entropy_loss_weight > 0:
                returns["attention_entropy_loss"] = (
                    0.5 * (attack_entropy + move_entropy)
                ) * self.attention_entropy_loss_weight
            move_entropy_gap = None
            if self.move_entropy_target_loss_weight > 0:
                target_entropy = move_entropy.new_tensor(float(self.move_entropy_target))
                move_entropy_gap = move_entropy - target_entropy
                returns["move_selective_entropy_loss"] = (
                    (self.move_entropy_target_loss_weight * move_comm_factor)
                    * move_entropy_gap.pow(2)
                )
            move_no_comm_gap = None
            if self.move_no_comm_target_loss_weight > 0:
                target_no_comm = move_no_comm_prob.new_tensor(
                    float(self.move_no_comm_target)
                )
                move_no_comm_gap = move_no_comm_prob - target_no_comm
                returns["move_no_comm_loss"] = (
                    (self.move_no_comm_target_loss_weight * move_comm_factor)
                    * move_no_comm_gap.clamp(min=0.0).pow(2)
                )
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

        return (
            self.comm_warmup_start_factor
            + (self.comm_warmup_end_factor - self.comm_warmup_start_factor) * progress
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
        if move_entropy_gap is not None:
            logs["Scalar_targeted_move_entropy_gap"] = move_entropy_gap.detach()
        if self.move_no_comm_target is not None:
            logs["Scalar_targeted_move_no_comm_target"] = th.tensor(
                float(self.move_no_comm_target), device=move_alpha.device
            )
        logs["Scalar_targeted_move_no_comm_target_loss_weight"] = th.tensor(
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

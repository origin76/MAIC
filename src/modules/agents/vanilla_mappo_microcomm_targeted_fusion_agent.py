import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VanillaMAPPOMicroCommTargetedFusionAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOMicroCommTargetedFusionAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.attention_heads = getattr(args, "attention_heads", 1)
        self.attention_dim = getattr(args, "attention_dim", 16)
        self.comm_value_dim = getattr(args, "comm_value_dim", 8)
        self.topk = getattr(args, "comm_topk", 1)
        self.log_attention_maps = getattr(args, "log_attention_maps", False)
        self.comm_detach_backbone = getattr(args, "comm_detach_backbone", False)
        self.comm_lr_multiplier = getattr(args, "comm_lr_multiplier", 1.0)
        self.attention_entropy_loss_weight = getattr(args, "attention_entropy_loss_weight", 0.0)

        self.semantic_action_offset = getattr(args, "semantic_action_offset", 6)
        self.relation_feature_indices = list(getattr(args, "relation_feature_indices", [0, 1, 2, 3, 4]))
        self.sender_state_dim = getattr(args, "targeted_sender_state_dim", 2)
        self.use_no_comm_token = getattr(args, "use_no_comm_token", True)
        self.attack_fusion_scale = getattr(args, "attack_fusion_scale", 0.1)
        self.attack_gate_init_bias = getattr(args, "attack_gate_init_bias", -2.5)
        self.attack_gate_floor = getattr(args, "attack_gate_floor", 0.0)
        self.attack_delta_zero_init = getattr(args, "attack_delta_zero_init", True)
        self.intent_detach = getattr(args, "intent_detach", False)
        self.intent_mask_unavailable = getattr(args, "intent_mask_unavailable", True)

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if self.semantic_action_offset >= self.n_actions:
            raise ValueError("semantic_action_offset must be smaller than n_actions")
        if len(self.relation_feature_indices) == 0:
            raise ValueError("relation_feature_indices must be non-empty")
        if self.sender_state_dim != 2:
            raise ValueError("targeted_sender_state_dim is currently fixed to 2")

        self.head_dim = self.attention_dim // self.attention_heads
        self.attack_action_dim = self.n_actions - self.semantic_action_offset
        hidden_dim = args.rnn_hidden_dim
        flat_comm_dim = self.attention_heads * self.comm_value_dim
        relation_dim = len(self.relation_feature_indices)

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )

        self.query_proj = nn.Linear(hidden_dim, self.attention_heads * self.head_dim)
        self.key_pair_proj = nn.Linear(hidden_dim + relation_dim, self.attention_heads * self.head_dim)
        self.value_proj = nn.Linear(self.attack_action_dim + self.sender_state_dim, self.attention_heads * self.comm_value_dim)
        self.message_layer_norm = nn.LayerNorm(flat_comm_dim)

        attack_gate_hidden_dim = getattr(args, "attack_gate_hidden_dim", hidden_dim)
        attack_delta_hidden_dim = getattr(args, "attack_delta_hidden_dim", hidden_dim)
        fusion_input_dim = hidden_dim + flat_comm_dim
        self.attack_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, attack_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(attack_gate_hidden_dim, 1),
        )
        self.attack_delta_head = nn.Sequential(
            nn.Linear(fusion_input_dim, attack_delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(attack_delta_hidden_dim, self.attack_action_dim),
        )

        if self.use_no_comm_token:
            self.null_key = nn.Parameter(th.zeros(1, 1, 1, self.attention_heads, self.head_dim))
            self.null_value = nn.Parameter(th.zeros(1, 1, 1, self.attention_heads, self.comm_value_dim))

        self._comm_param_prefixes = (
            "query_proj",
            "key_pair_proj",
            "value_proj",
            "message_layer_norm",
            "attack_gate",
            "attack_delta_head",
            "null_key",
            "null_value",
        )

        nn.init.constant_(self.attack_gate[-1].bias, self.attack_gate_init_bias)
        if self.attack_delta_zero_init:
            nn.init.constant_(self.attack_delta_head[-1].weight, 0.0)
            nn.init.constant_(self.attack_delta_head[-1].bias, 0.0)

        self._obs_layout_ready = False
        self.ally_feat_start = None
        self.ally_feat_end = None
        self.ally_feat_dim = None

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        agent_hidden = h.reshape(bs, self.n_agents, -1)

        raw_obs = kwargs.get("raw_obs", None)
        avail_actions = kwargs.get("avail_actions", None)
        comm_source = agent_hidden.detach() if self.comm_detach_backbone else agent_hidden

        local_logits = self.policy_head(agent_hidden.reshape(bs * self.n_agents, -1)).reshape(bs, self.n_agents, self.n_actions)
        local_base_logits = local_logits[:, :, :self.semantic_action_offset]
        local_attack_logits = local_logits[:, :, self.semantic_action_offset:]
        attack_avail_actions = self._get_attack_avail_actions(avail_actions)
        attack_probs = self._build_attack_probs(local_attack_logits, attack_avail_actions)
        attack_top1_mass = attack_probs.max(dim=-1, keepdim=True)[0]
        if attack_avail_actions is None:
            can_attack = attack_probs.new_ones(bs, self.n_agents, 1)
        else:
            can_attack = (attack_avail_actions.sum(dim=-1, keepdim=True) > 0).float()

        value_source = th.cat([attack_probs, can_attack, attack_top1_mass], dim=-1)
        if self.intent_detach:
            value_source = value_source.detach()
        sender_values = self.value_proj(value_source).reshape(bs, self.n_agents, self.attention_heads, self.comm_value_dim)

        relation_features = self._extract_relation_features(raw_obs, bs, agent_hidden.device)
        sender_hidden = comm_source.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        pair_input = th.cat([sender_hidden, relation_features], dim=-1)

        queries = self.query_proj(comm_source).reshape(bs, self.n_agents, self.attention_heads, self.head_dim)
        pair_keys = self.key_pair_proj(pair_input.reshape(bs * self.n_agents * self.n_agents, -1)).reshape(
            bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim
        )

        scores = (queries.unsqueeze(2) * pair_keys).sum(dim=-1) / (self.head_dim ** 0.5)
        self_mask = th.eye(self.n_agents, device=scores.device, dtype=th.bool).view(1, self.n_agents, self.n_agents, 1)
        scores = scores.masked_fill(self_mask, -1e10)

        if self.use_no_comm_token:
            null_scores = (queries.unsqueeze(2) * self.null_key).sum(dim=-1) / (self.head_dim ** 0.5)
            scores = th.cat([scores, null_scores], dim=2)

        sparse_scores = self._apply_topk_mask(scores)
        alpha = F.softmax(sparse_scores, dim=2)

        expanded_values = sender_values.unsqueeze(1).expand(-1, self.n_agents, -1, -1, -1)
        if self.use_no_comm_token:
            null_values = self.null_value.expand(bs, self.n_agents, -1, -1, -1)
            expanded_values = th.cat([expanded_values, null_values], dim=2)

        head_messages = (alpha.unsqueeze(-1) * expanded_values).sum(dim=2)
        flat_messages = head_messages.reshape(bs, self.n_agents, -1)
        normed_messages = self.message_layer_norm(flat_messages)

        fusion_input = th.cat([agent_hidden, normed_messages], dim=-1)
        raw_attack_gate = th.sigmoid(self.attack_gate(fusion_input.reshape(bs * self.n_agents, -1))).reshape(bs, self.n_agents, 1)
        attack_gate = self.attack_gate_floor + (1.0 - self.attack_gate_floor) * raw_attack_gate
        attack_delta = self.attack_delta_head(fusion_input.reshape(bs * self.n_agents, -1)).reshape(bs, self.n_agents, self.attack_action_dim)
        fused_attack_logits = local_attack_logits + self.attack_fusion_scale * attack_gate * attack_delta

        final_logits = th.cat([local_base_logits, fused_attack_logits], dim=-1)

        returns = {}
        if kwargs.get("train_mode", False):
            attention_entropy = self._compute_mean_attention_entropy(alpha)
            if self.attention_entropy_loss_weight > 0:
                returns["attention_entropy_loss"] = attention_entropy * self.attention_entropy_loss_weight
            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(
                    alpha,
                    relation_features,
                    normed_messages,
                    attack_gate,
                    raw_attack_gate,
                    attack_delta,
                    attack_probs,
                )

        return final_logits.reshape(bs * self.n_agents, self.n_actions), h, returns

    def _get_attack_avail_actions(self, avail_actions):
        if avail_actions is None:
            return None
        return avail_actions[:, :, self.semantic_action_offset:]

    def _build_attack_probs(self, attack_logits, attack_avail_actions):
        if attack_avail_actions is None or not self.intent_mask_unavailable:
            return F.softmax(attack_logits, dim=-1)

        masked_logits = attack_logits.masked_fill(attack_avail_actions == 0, -1e10)
        attack_probs = F.softmax(masked_logits, dim=-1)
        valid_mask = (attack_avail_actions.sum(dim=-1, keepdim=True) > 0).float()
        return attack_probs * valid_mask

    def _extract_relation_features(self, raw_obs, batch_size, device):
        relation_dim = len(self.relation_feature_indices)
        if raw_obs is None:
            return th.zeros(batch_size, self.n_agents, self.n_agents, relation_dim, device=device)

        self._maybe_init_obs_layout(raw_obs.size(-1))
        ally_flat = raw_obs[:, :, self.ally_feat_start:self.ally_feat_end]
        ally_feats = ally_flat.reshape(raw_obs.size(0), self.n_agents, self.n_agents - 1, self.ally_feat_dim)
        relation = raw_obs.new_zeros(raw_obs.size(0), self.n_agents, self.n_agents, relation_dim)

        for receiver_idx in range(self.n_agents):
            sender_indices = [ally_idx for ally_idx in range(self.n_agents) if ally_idx != receiver_idx]
            relation[:, receiver_idx, sender_indices] = ally_feats[:, receiver_idx, :, self.relation_feature_indices]

        return relation

    def _maybe_init_obs_layout(self, obs_shape):
        if self._obs_layout_ready:
            return

        env_args = getattr(self.args, "env_args", {})
        move_feats_dim = env_args.get("n_actions_move", 4)
        last_action_dim = self.n_actions if env_args.get("obs_last_action", False) else 0
        n_enemies = self.n_actions - self.semantic_action_offset
        numerator = obs_shape - move_feats_dim + 4 + last_action_dim
        denominator = n_enemies + self.n_agents

        if denominator <= 0 or numerator % denominator != 0:
            raise ValueError(
                "Unable to infer ally feature dim from obs_shape={} n_agents={} n_enemies={}".format(
                    obs_shape,
                    self.n_agents,
                    n_enemies,
                )
            )

        self.ally_feat_dim = numerator // denominator
        enemy_feat_dim = self.ally_feat_dim
        self.ally_feat_start = move_feats_dim + n_enemies * enemy_feat_dim
        self.ally_feat_end = self.ally_feat_start + (self.n_agents - 1) * self.ally_feat_dim

        if self.ally_feat_end > obs_shape:
            raise ValueError("Inferred ally feature slice exceeds obs dim")
        if max(self.relation_feature_indices) >= self.ally_feat_dim:
            raise ValueError(
                "relation_feature_indices exceed inferred ally feature dim {}".format(self.ally_feat_dim)
            )

        self._obs_layout_ready = True

    def _apply_topk_mask(self, scores):
        max_options = scores.size(2)
        k = min(max(1, self.topk), max_options)
        if k >= max_options:
            return scores

        scores_perm = scores.permute(0, 1, 3, 2)
        topk_indices = scores_perm.topk(k=k, dim=-1).indices
        keep_mask = th.zeros_like(scores_perm, dtype=th.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        scores_perm = scores_perm.masked_fill(~keep_mask, -1e10)
        return scores_perm.permute(0, 1, 3, 2)

    def _compute_mean_attention_entropy(self, alpha):
        detached_alpha = th.clamp(alpha, min=1e-8)
        return -(detached_alpha * th.log(detached_alpha)).sum(dim=2).mean()

    def get_actor_optim_groups(self, base_lr):
        if self.comm_lr_multiplier <= 1.0:
            return [{
                "params": list(self.parameters()),
                "lr": base_lr,
                "initial_lr": base_lr,
                "group_name": "actor",
            }]

        comm_params = []
        backbone_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(self._comm_param_prefixes):
                comm_params.append(param)
            else:
                backbone_params.append(param)

        if len(comm_params) == 0 or len(backbone_params) == 0:
            return [{
                "params": list(self.parameters()),
                "lr": base_lr,
                "initial_lr": base_lr,
                "group_name": "actor",
            }]

        comm_lr = base_lr * self.comm_lr_multiplier
        return [
            {
                "params": backbone_params,
                "lr": base_lr,
                "initial_lr": base_lr,
                "group_name": "actor_backbone",
            },
            {
                "params": comm_params,
                "lr": comm_lr,
                "initial_lr": comm_lr,
                "group_name": "actor_comm",
            },
        ]

    def build_logging_payload(
        self,
        alpha,
        relation_features,
        normed_messages,
        attack_gate,
        raw_attack_gate,
        attack_delta,
        attack_probs,
    ):
        logs = {}
        detached_alpha = alpha.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-8) * th.log(th.clamp(detached_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))

        logs["Scalar_targeted_mean_attn_entropy"] = head_entropy.mean()
        logs["Scalar_targeted_attack_gate_mean"] = attack_gate.detach().mean()
        logs["Scalar_targeted_attack_gate_raw_mean"] = raw_attack_gate.detach().mean()
        logs["Scalar_targeted_attack_delta_norm"] = attack_delta.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_attack_delta_abs_mean"] = attack_delta.detach().abs().mean()
        logs["Scalar_targeted_message_norm"] = normed_messages.detach().norm(dim=-1).mean()
        logs["Scalar_targeted_no_comm_prob"] = detached_alpha[:, :, -1, :].mean() if self.use_no_comm_token else th.tensor(0.0, device=alpha.device)
        logs["Scalar_targeted_relation_visible_ratio"] = relation_features[:, :, :, 0].detach().mean()
        logs["Scalar_targeted_attack_intent_top1_mass"] = attack_probs.detach().max(dim=-1)[0].mean()
        logs["Scalar_targeted_edge_budget_ratio"] = th.tensor(
            float(min(max(1, self.topk), alpha.size(2))) / float(alpha.size(2)),
            device=alpha.device,
        )
        logs["Scalar_message_norm"] = logs["Scalar_targeted_message_norm"]

        for head_idx in range(self.attention_heads):
            logs["Scalar_targeted_head_{}_entropy".format(head_idx)] = head_entropy[head_idx]
            if self.log_attention_maps:
                logs["Histogram_targeted_head_{}_attention".format(head_idx)] = detached_alpha[:, :, :, head_idx]

        return logs

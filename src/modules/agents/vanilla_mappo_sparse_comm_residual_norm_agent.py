import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VanillaMAPPOSparseCommResidualNormAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOSparseCommResidualNormAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.attention_heads = getattr(args, "attention_heads", 2)
        self.attention_dim = getattr(args, "attention_dim", 32)
        self.comm_value_dim = getattr(args, "comm_value_dim", 4)
        self.topk = getattr(args, "comm_topk", max(1, self.n_agents - 1))
        self.head_budget_loss_weight = getattr(args, "head_budget_loss_weight", 0.0)
        self.log_attention_maps = getattr(args, "log_attention_maps", False)
        self.comm_eval_threshold = getattr(args, "comm_eval_threshold", 0.25)
        self.comm_eval_renorm = getattr(args, "comm_eval_renorm", True)
        self.residual_comm_scale = getattr(args, "residual_comm_scale", 0.5)

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        self.head_dim = self.attention_dim // self.attention_heads

        hidden_dim = args.rnn_hidden_dim
        flat_comm_dim = self.attention_heads * self.comm_value_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, self.attention_heads * self.head_dim)
        self.key_proj = nn.Linear(hidden_dim, self.attention_heads * self.head_dim)
        self.value_proj = nn.Linear(hidden_dim, self.attention_heads * self.comm_value_dim)
        self.head_gate = nn.Linear(hidden_dim, self.attention_heads)

        self.message_layer_norm = nn.LayerNorm(self.comm_value_dim)
        self.comm_fusion_norm = nn.LayerNorm(flat_comm_dim)
        self.residual_proj = nn.Linear(flat_comm_dim, hidden_dim)

        residual_gate_hidden_dim = getattr(args, "residual_gate_hidden_dim", hidden_dim)
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim + flat_comm_dim, residual_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(residual_gate_hidden_dim, 1),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.n_actions),
        )

        head_gate_init_bias = getattr(args, "head_gate_init_bias", -0.5)
        residual_gate_init_bias = getattr(args, "residual_gate_init_bias", -1.0)
        nn.init.constant_(self.head_gate.bias, head_gate_init_bias)
        nn.init.constant_(self.residual_gate[-1].bias, residual_gate_init_bias)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        agent_hidden = h.reshape(bs, self.n_agents, -1)

        queries = self.query_proj(agent_hidden).reshape(bs, self.n_agents, self.attention_heads, self.head_dim)
        keys = self.key_proj(agent_hidden).reshape(bs, self.n_agents, self.attention_heads, self.head_dim)
        values = self.value_proj(agent_hidden).reshape(bs, self.n_agents, self.attention_heads, self.comm_value_dim)

        scores = (queries.unsqueeze(2) * keys.unsqueeze(1)).sum(dim=-1) / (self.head_dim ** 0.5)
        self_mask = th.eye(self.n_agents, device=scores.device, dtype=th.bool).view(1, self.n_agents, self.n_agents, 1)
        scores = scores.masked_fill(self_mask, -1e10)
        sparse_scores = self._apply_topk_mask(scores)
        alpha = F.softmax(sparse_scores, dim=2)

        if test_mode:
            threshold = self.comm_eval_threshold / max(1, self.n_agents - 1)
            pruned_alpha = alpha.masked_fill(alpha < threshold, 0.0)
            if self.comm_eval_renorm:
                denom = pruned_alpha.sum(dim=2, keepdim=True)
                renorm_alpha = pruned_alpha / denom.clamp(min=1e-8)
                alpha = th.where((denom > 0).expand_as(alpha), renorm_alpha, alpha)
            else:
                alpha = pruned_alpha

        raw_head_messages = (alpha.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        normed_head_messages = self.message_layer_norm(raw_head_messages)
        head_gates = th.sigmoid(self.head_gate(agent_hidden)).unsqueeze(-1)
        gated_head_messages = normed_head_messages * head_gates

        flat_messages = gated_head_messages.reshape(bs, self.n_agents, -1)
        flat_messages = self.comm_fusion_norm(flat_messages)

        residual_gate_input = th.cat([agent_hidden, flat_messages], dim=-1)
        residual_gate = th.sigmoid(
            self.residual_gate(residual_gate_input.reshape(bs * self.n_agents, -1))
        ).reshape(bs, self.n_agents, 1)

        residual_update = self.residual_proj(flat_messages.reshape(bs * self.n_agents, -1)).reshape(bs, self.n_agents, -1)
        residual_update = residual_update * (self.residual_comm_scale * residual_gate)

        fused_hidden = agent_hidden + residual_update
        logits = self.policy_head(fused_hidden.reshape(bs * self.n_agents, -1))

        returns = {}
        if kwargs.get("train_mode", False):
            returns["budget_loss"] = self.calculate_budget_loss(head_gates)
            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(
                    alpha,
                    head_gates,
                    raw_head_messages,
                    gated_head_messages,
                    residual_gate,
                    residual_update,
                )

        return logits, h, returns

    def _apply_topk_mask(self, scores):
        max_neighbors = max(1, self.n_agents - 1)
        k = min(max(1, self.topk), max_neighbors)
        if k >= max_neighbors:
            return scores

        scores_perm = scores.permute(0, 1, 3, 2)
        topk_indices = scores_perm.topk(k=k, dim=-1).indices
        keep_mask = th.zeros_like(scores_perm, dtype=th.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        scores_perm = scores_perm.masked_fill(~keep_mask, -1e10)
        return scores_perm.permute(0, 1, 3, 2)

    def calculate_budget_loss(self, head_gates):
        if self.head_budget_loss_weight <= 0:
            return head_gates.new_zeros(())
        return head_gates.mean() * self.head_budget_loss_weight

    def build_logging_payload(
        self,
        alpha,
        head_gates,
        raw_head_messages,
        gated_head_messages,
        residual_gate,
        residual_update,
    ):
        logs = {}
        detached_alpha = alpha.detach()
        detached_gates = head_gates.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-8) * th.log(th.clamp(detached_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))

        logs["Scalar_mean_attn_entropy"] = head_entropy.mean()
        logs["Scalar_mean_head_gate"] = detached_gates.mean()
        logs["Scalar_message_norm"] = gated_head_messages.detach().norm(dim=-1).mean()
        logs["Scalar_raw_message_norm"] = raw_head_messages.detach().norm(dim=-1).mean()
        logs["Scalar_residual_gate"] = residual_gate.detach().mean()
        logs["Scalar_residual_norm"] = residual_update.detach().norm(dim=-1).mean()
        logs["Scalar_edge_budget_ratio"] = th.tensor(
            float(min(max(1, self.topk), max(1, self.n_agents - 1))) / float(max(1, self.n_agents - 1)),
            device=alpha.device,
        )

        for head_idx in range(self.attention_heads):
            logs["Scalar_head_{}_entropy".format(head_idx)] = head_entropy[head_idx]
            logs["Scalar_head_{}_gate".format(head_idx)] = detached_gates[:, :, head_idx, :].mean()
            if self.log_attention_maps:
                logs["Histogram_head_{}_attention".format(head_idx)] = detached_alpha[:, :, :, head_idx]

        return logs

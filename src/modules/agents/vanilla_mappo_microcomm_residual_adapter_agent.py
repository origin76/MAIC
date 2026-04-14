import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VanillaMAPPOMicroCommResidualAdapterAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOMicroCommResidualAdapterAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.attention_heads = getattr(args, "attention_heads", 1)
        self.attention_dim = getattr(args, "attention_dim", 16)
        self.comm_value_dim = getattr(args, "comm_value_dim", 4)
        self.topk = getattr(args, "comm_topk", max(1, self.n_agents - 1))
        self.log_attention_maps = getattr(args, "log_attention_maps", False)
        self.residual_comm_scale = getattr(args, "residual_comm_scale", 0.1)
        self.comm_detach_backbone = getattr(args, "comm_detach_backbone", True)
        self.adapter_gate_init_bias = getattr(args, "adapter_gate_init_bias", -2.5)
        self.adapter_zero_init = getattr(args, "adapter_zero_init", True)
        self.attention_entropy_loss_weight = getattr(args, "attention_entropy_loss_weight", 0.0)
        self.comm_lr_multiplier = getattr(args, "comm_lr_multiplier", 1.0)

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

        self.comm_layer_norm = nn.LayerNorm(flat_comm_dim)

        adapter_gate_hidden_dim = getattr(args, "adapter_gate_hidden_dim", hidden_dim)
        self.adapter_gate = nn.Sequential(
            nn.Linear(hidden_dim + flat_comm_dim, adapter_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_gate_hidden_dim, 1),
        )
        self.residual_proj = nn.Linear(flat_comm_dim, hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.n_actions),
        )

        self._comm_param_prefixes = (
            "query_proj",
            "key_proj",
            "value_proj",
            "comm_layer_norm",
            "adapter_gate",
            "residual_proj",
        )

        nn.init.constant_(self.adapter_gate[-1].bias, self.adapter_gate_init_bias)
        if self.adapter_zero_init:
            nn.init.constant_(self.residual_proj.weight, 0.0)
            nn.init.constant_(self.residual_proj.bias, 0.0)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        agent_hidden = h.reshape(bs, self.n_agents, -1)

        comm_source = agent_hidden.detach() if self.comm_detach_backbone else agent_hidden

        queries = self.query_proj(comm_source).reshape(bs, self.n_agents, self.attention_heads, self.head_dim)
        keys = self.key_proj(comm_source).reshape(bs, self.n_agents, self.attention_heads, self.head_dim)
        values = self.value_proj(comm_source).reshape(bs, self.n_agents, self.attention_heads, self.comm_value_dim)

        scores = (queries.unsqueeze(2) * keys.unsqueeze(1)).sum(dim=-1) / (self.head_dim ** 0.5)
        self_mask = th.eye(self.n_agents, device=scores.device, dtype=th.bool).view(1, self.n_agents, self.n_agents, 1)
        scores = scores.masked_fill(self_mask, -1e10)
        sparse_scores = self._apply_topk_mask(scores)
        alpha = F.softmax(sparse_scores, dim=2)

        head_messages = (alpha.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        flat_messages = head_messages.reshape(bs, self.n_agents, -1)
        normed_messages = self.comm_layer_norm(flat_messages)

        gate_input = th.cat([comm_source, normed_messages], dim=-1)
        adapter_gate = th.sigmoid(
            self.adapter_gate(gate_input.reshape(bs * self.n_agents, -1))
        ).reshape(bs, self.n_agents, 1)

        residual_update = self.residual_proj(normed_messages.reshape(bs * self.n_agents, -1)).reshape(bs, self.n_agents, -1)
        residual_update = residual_update * (self.residual_comm_scale * adapter_gate)

        fused_hidden = agent_hidden + residual_update
        logits = self.policy_head(fused_hidden.reshape(bs * self.n_agents, -1))

        returns = {}
        if kwargs.get("train_mode", False):
            attention_entropy = self._compute_mean_attention_entropy(alpha)
            if self.attention_entropy_loss_weight > 0:
                returns["attention_entropy_loss"] = attention_entropy * self.attention_entropy_loss_weight
            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(
                    alpha,
                    flat_messages,
                    normed_messages,
                    adapter_gate,
                    residual_update,
                    attention_entropy,
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

    def build_logging_payload(self, alpha, flat_messages, normed_messages, adapter_gate, residual_update, attention_entropy):
        logs = {}
        detached_alpha = alpha.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-8) * th.log(th.clamp(detached_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))

        logs["Scalar_mean_attn_entropy"] = head_entropy.mean()
        logs["Scalar_attention_entropy_raw"] = attention_entropy.detach()
        logs["Scalar_attention_entropy_loss_weight"] = th.tensor(
            float(self.attention_entropy_loss_weight), device=alpha.device
        )
        logs["Scalar_adapter_gate_mean"] = adapter_gate.detach().mean()
        logs["Scalar_raw_message_norm"] = flat_messages.detach().norm(dim=-1).mean()
        logs["Scalar_adapter_message_norm"] = normed_messages.detach().norm(dim=-1).mean()
        logs["Scalar_adapter_residual_norm"] = residual_update.detach().norm(dim=-1).mean()
        logs["Scalar_message_norm"] = logs["Scalar_adapter_message_norm"]
        logs["Scalar_residual_norm"] = logs["Scalar_adapter_residual_norm"]
        logs["Scalar_edge_budget_ratio"] = th.tensor(
            float(min(max(1, self.topk), max(1, self.n_agents - 1))) / float(max(1, self.n_agents - 1)),
            device=alpha.device,
        )

        for head_idx in range(self.attention_heads):
            logs["Scalar_head_{}_entropy".format(head_idx)] = head_entropy[head_idx]
            if self.log_attention_maps:
                logs["Histogram_head_{}_attention".format(head_idx)] = detached_alpha[:, :, :, head_idx]

        return logs

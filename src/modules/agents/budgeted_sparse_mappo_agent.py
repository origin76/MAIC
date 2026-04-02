import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BudgetedSparseMAPPOAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(BudgetedSparseMAPPOAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.attention_heads = getattr(args, "attention_heads", 2)
        self.attention_dim = args.attention_dim
        self.comm_value_dim = getattr(args, "comm_value_dim", 16)
        self.topk = getattr(args, "comm_topk", max(1, self.n_agents - 1))
        self.head_budget_loss_weight = getattr(args, "head_budget_loss_weight", 0.0)
        self.log_attention_maps = getattr(args, "log_attention_maps", False)

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        self.head_dim = self.attention_dim // self.attention_heads

        hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, self.attention_heads * self.head_dim)
        self.key_proj = nn.Linear(hidden_dim, self.attention_heads * self.head_dim)
        self.value_proj = nn.Linear(hidden_dim, self.attention_heads * self.comm_value_dim)
        self.head_gate = nn.Linear(hidden_dim, self.attention_heads)

        actor_input_dim = hidden_dim + self.attention_heads * self.comm_value_dim
        self.policy_head = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.n_actions)
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        agent_hidden = h.view(bs, self.n_agents, -1)

        queries = self.query_proj(agent_hidden).view(bs, self.n_agents, self.attention_heads, self.head_dim)
        keys = self.key_proj(agent_hidden).view(bs, self.n_agents, self.attention_heads, self.head_dim)
        values = self.value_proj(agent_hidden).view(bs, self.n_agents, self.attention_heads, self.comm_value_dim)

        scores = (queries.unsqueeze(2) * keys.unsqueeze(1)).sum(dim=-1) / (self.head_dim ** 0.5)
        self_mask = th.eye(self.n_agents, device=scores.device, dtype=th.bool).view(1, self.n_agents, self.n_agents, 1)
        scores = scores.masked_fill(self_mask, -1e10)
        sparse_scores = self._apply_topk_mask(scores)
        alpha = F.softmax(sparse_scores, dim=2)

        if test_mode:
            alpha = alpha.masked_fill(alpha < (0.25 / max(1, self.n_agents - 1)), 0.0)

        head_messages = (alpha.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        head_gates = th.sigmoid(self.head_gate(agent_hidden)).unsqueeze(-1)
        gated_messages = head_messages * head_gates

        policy_input = th.cat([agent_hidden, gated_messages.reshape(bs, self.n_agents, -1)], dim=-1)
        logits = self.policy_head(policy_input.reshape(bs * self.n_agents, -1))

        returns = {}
        if kwargs.get("train_mode", False):
            returns["budget_loss"] = self.calculate_budget_loss(head_gates)
            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(alpha, head_gates, gated_messages)

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

    def build_logging_payload(self, alpha, head_gates, gated_messages):
        logs = {}
        detached_alpha = alpha.detach()
        detached_gates = head_gates.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-8) * th.log(th.clamp(detached_alpha, min=1e-8))
        ).sum(dim=2).mean(dim=(0, 1))

        logs["Scalar_mean_attn_entropy"] = head_entropy.mean()
        logs["Scalar_mean_head_gate"] = detached_gates.mean()
        logs["Scalar_message_norm"] = gated_messages.detach().norm(dim=-1).mean()
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

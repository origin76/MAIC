import torch as th
import torch.nn.functional as F

from .budgeted_sparse_mappo_agent import BudgetedSparseMAPPOAgent


class BudgetedSparseMAPPOEvalNormAgent(BudgetedSparseMAPPOAgent):
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
            threshold = 0.25 / max(1, self.n_agents - 1)
            pruned_alpha = alpha.masked_fill(alpha < threshold, 0.0)
            denom = pruned_alpha.sum(dim=2, keepdim=True)
            renorm_alpha = pruned_alpha / denom.clamp(min=1e-8)
            alpha = th.where((denom > 0).expand_as(alpha), renorm_alpha, alpha)

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

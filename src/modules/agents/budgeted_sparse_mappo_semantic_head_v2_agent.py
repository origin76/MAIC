import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .budgeted_sparse_mappo_agent import BudgetedSparseMAPPOAgent


class BudgetedSparseMAPPOSemanticHeadV2Agent(BudgetedSparseMAPPOAgent):
    def __init__(self, input_shape, args):
        super(BudgetedSparseMAPPOSemanticHeadV2Agent, self).__init__(input_shape, args)

        self.semantic_action_head_idx = getattr(args, "semantic_action_head_idx", 0)
        self.semantic_state_head_idx = getattr(args, "semantic_state_head_idx", 1)
        self.semantic_action_offset = getattr(args, "semantic_action_offset", 6)
        shared_aux_hidden_dim = getattr(args, "semantic_aux_hidden_dim", max(self.comm_value_dim * 2, 16))
        action_aux_hidden_dim = getattr(args, "semantic_action_aux_hidden_dim", shared_aux_hidden_dim)
        state_aux_hidden_dim = getattr(args, "semantic_state_aux_hidden_dim", shared_aux_hidden_dim)
        self.semantic_state_dim = getattr(args, "semantic_state_dim", 5)

        if self.attention_heads < 2:
            raise ValueError("semantic_head_v2 requires at least 2 attention heads")
        if self.semantic_action_head_idx >= self.attention_heads:
            raise ValueError("semantic_action_head_idx exceeds attention_heads")
        if self.semantic_state_head_idx >= self.attention_heads:
            raise ValueError("semantic_state_head_idx exceeds attention_heads")
        if self.semantic_action_head_idx == self.semantic_state_head_idx:
            raise ValueError("semantic_head_v2 requires different action/state head indices")

        self.semantic_action_dim = self.n_actions - self.semantic_action_offset
        if self.semantic_action_dim <= 0:
            raise ValueError("semantic_action_offset must be smaller than n_actions")
        if self.semantic_state_dim <= 0:
            raise ValueError("semantic_state_dim must be positive")

        self.semantic_action_predictor = nn.Sequential(
            nn.Linear(self.comm_value_dim, action_aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_aux_hidden_dim, self.semantic_action_dim),
        )
        self.semantic_state_predictor = nn.Sequential(
            nn.Linear(self.comm_value_dim, state_aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_aux_hidden_dim, self.semantic_state_dim),
        )

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
            action_alpha = alpha[:, :, :, self.semantic_action_head_idx].detach()
            action_values = values[:, :, self.semantic_action_head_idx, :]
            action_message = th.matmul(action_alpha, action_values)
            semantic_action_pred = th.sigmoid(
                self.semantic_action_predictor(action_message.reshape(bs * self.n_agents, -1))
            ).view(bs, self.n_agents, self.semantic_action_dim)

            state_alpha = alpha[:, :, :, self.semantic_state_head_idx].detach()
            state_values = values[:, :, self.semantic_state_head_idx, :]
            state_message = th.matmul(state_alpha, state_values)
            semantic_state_pred = self.semantic_state_predictor(
                state_message.reshape(bs * self.n_agents, -1)
            ).view(bs, self.n_agents, self.semantic_state_dim)

            returns["budget_loss"] = self.calculate_budget_loss(head_gates)
            returns["semantic_action_pred"] = semantic_action_pred
            returns["semantic_action_attn"] = action_alpha
            returns["semantic_state_pred"] = semantic_state_pred
            returns["semantic_state_attn"] = state_alpha

            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(
                    alpha,
                    head_gates,
                    gated_messages,
                    action_message,
                    state_message,
                )

        return logits, h, returns

    def build_logging_payload(self, alpha, head_gates, gated_messages, action_message, state_message):
        logs = super().build_logging_payload(alpha, head_gates, gated_messages)
        logs["Scalar_semantic_action_head_message_norm"] = action_message.detach().norm(dim=-1).mean()
        logs["Scalar_semantic_state_head_message_norm"] = state_message.detach().norm(dim=-1).mean()
        return logs

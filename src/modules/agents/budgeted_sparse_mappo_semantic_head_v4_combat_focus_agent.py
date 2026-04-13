import torch as th
import torch.nn.functional as F

from .budgeted_sparse_mappo_semantic_head_v3_integration_agent import BudgetedSparseMAPPOSemanticHeadV3IntegrationAgent


class BudgetedSparseMAPPOSemanticHeadV4CombatFocusAgent(BudgetedSparseMAPPOSemanticHeadV3IntegrationAgent):
    def __init__(self, input_shape, args):
        super(BudgetedSparseMAPPOSemanticHeadV4CombatFocusAgent, self).__init__(input_shape, args)
        self.action_message_scale = getattr(args, "action_message_scale", 1.2)

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

        raw_head_messages = (alpha.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)
        head_messages = self.message_layer_norm(raw_head_messages)

        repeated_hidden = agent_hidden.unsqueeze(2).expand(-1, -1, self.attention_heads, -1)
        gate_inputs = th.cat([repeated_hidden, head_messages], dim=-1)
        head_gates = th.sigmoid(
            self.integration_head_gate(gate_inputs.reshape(bs * self.n_agents * self.attention_heads, -1))
        ).view(bs, self.n_agents, self.attention_heads, 1)
        gated_messages = head_messages * head_gates

        gated_messages[:, :, self.semantic_action_head_idx, :] = (
            gated_messages[:, :, self.semantic_action_head_idx, :] * self.action_message_scale
        )

        policy_input = th.cat([agent_hidden, gated_messages.reshape(bs, self.n_agents, -1)], dim=-1)
        logits = self.policy_head(policy_input.reshape(bs * self.n_agents, -1))

        returns = {}
        if kwargs.get("train_mode", False):
            action_alpha = alpha[:, :, :, self.semantic_action_head_idx].detach()
            action_values = values[:, :, self.semantic_action_head_idx, :]
            action_message = self.message_layer_norm(th.matmul(action_alpha, action_values))
            semantic_action_pred = th.sigmoid(
                self.semantic_action_predictor(action_message.reshape(bs * self.n_agents, -1))
            ).view(bs, self.n_agents, self.semantic_action_dim)

            state_alpha = alpha[:, :, :, self.semantic_state_head_idx].detach()
            state_values = values[:, :, self.semantic_state_head_idx, :]
            state_message = self.message_layer_norm(th.matmul(state_alpha, state_values))
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
                    raw_head_messages,
                    action_message,
                    state_message,
                )
                returns["logs"]["Scalar_action_message_scale"] = gated_messages.new_tensor(self.action_message_scale)

        return logits, h, returns

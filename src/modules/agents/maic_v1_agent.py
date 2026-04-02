import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions import kl_divergence


class MAICV1Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAICV1Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions
        self.attention_heads = max(1, getattr(args, "attention_heads", 2))
        self.attention_dim = args.attention_dim
        self.head_feature_dim = getattr(args, "head_feature_dim", 16)
        self.intent_head_idx = getattr(args, "intent_head_idx", 0)
        self.reward_head_idx = getattr(args, "reward_head_idx", 1)
        self.log_attention_maps = getattr(args, "log_attention_maps", False)

        if self.attention_heads < 2:
            raise ValueError("MAICV1Agent requires at least 2 attention heads")
        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if self.intent_head_idx == self.reward_head_idx:
            raise ValueError("intent_head_idx and reward_head_idx must be different")
        if max(self.intent_head_idx, self.reward_head_idx) >= self.attention_heads:
            raise ValueError("auxiliary head index exceeds number of attention heads")

        self.head_dim = self.attention_dim // self.attention_heads

        nn_hidden_size = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, nn_hidden_size),
            nn.BatchNorm1d(nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.n_agents * args.latent_dim * 2)
        )

        self.inference_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, nn_hidden_size),
            nn.BatchNorm1d(nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.latent_dim * 2)
        )

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.pair_feature_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.latent_dim, nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, self.attention_heads * self.head_feature_dim)
        )

        self.w_query = nn.Linear(args.rnn_hidden_dim, self.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, self.attention_dim)
        self.head_fusion_q = nn.Linear(self.attention_heads * self.head_feature_dim, args.n_actions)

        self.intent_predictor = nn.Sequential(
            nn.Linear(self.head_feature_dim, nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.n_actions)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.head_feature_dim, nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, 1)
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        latent_parameters = self.embed_net(h)
        latent_var = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor
        )
        latent_mean = latent_parameters[:, :self.n_agents * self.latent_dim]

        latent_mean = latent_mean.view(bs, self.n_agents, self.n_agents, self.latent_dim)
        latent_var = latent_var.view(bs, self.n_agents, self.n_agents, self.latent_dim)

        if test_mode:
            latent = latent_mean
        else:
            latent = D.Normal(latent_mean, latent_var ** 0.5).rsample()

        sender_hidden = h.view(bs, self.n_agents, -1)
        repeated_sender_hidden = sender_hidden.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        pair_input = th.cat([repeated_sender_hidden, latent], dim=-1)
        pair_features = self.pair_feature_net(pair_input.reshape(-1, pair_input.size(-1))).view(
            bs, self.n_agents, self.n_agents, self.attention_heads, self.head_feature_dim
        )

        query = self.w_query(sender_hidden).view(bs, self.n_agents, self.attention_heads, self.head_dim)
        key = self.w_key(latent).view(bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim)
        alpha = (query.unsqueeze(2) * key).sum(dim=-1) / (self.head_dim ** 0.5)
        self_mask = th.eye(self.n_agents, device=alpha.device, dtype=th.bool).view(
            1, self.n_agents, self.n_agents, 1
        )
        alpha = alpha.masked_fill(self_mask, -1e9)
        alpha = F.softmax(alpha, dim=2)

        if test_mode:
            alpha = alpha.masked_fill(alpha < (0.25 * 1 / self.n_agents), 0)

        head_messages = (alpha.unsqueeze(-1) * pair_features).sum(dim=1)
        comm_q = self.head_fusion_q(head_messages.reshape(bs * self.n_agents, -1))
        return_q = q + comm_q

        returns = {}
        if kwargs.get("train_mode", False):
            if hasattr(self.args, "mi_loss_weight") and self.args.mi_loss_weight > 0:
                returns["mi_loss"] = self.calculate_action_mi_loss(
                    sender_hidden, latent_mean, latent_var, return_q
                )
            if hasattr(self.args, "entropy_loss_weight") and self.args.entropy_loss_weight > 0:
                returns["entropy_loss"] = self.calculate_entropy_loss(alpha)

            intent_features = pair_features[:, :, :, self.intent_head_idx, :]
            reward_features = head_messages[:, :, self.reward_head_idx, :].mean(dim=1)
            returns["intent_logits"] = self.intent_predictor(
                intent_features.reshape(-1, self.head_feature_dim)
            ).view(bs, self.n_agents, self.n_agents, self.n_actions)
            returns["reward_pred"] = self.reward_predictor(reward_features)

            if kwargs.get("prepare_for_logging", False):
                returns["logs"] = self.build_logging_payload(alpha, head_messages)

        return return_q, h, returns

    def calculate_action_mi_loss(self, sender_hidden, latent_mean, latent_var, q):
        g1 = D.Normal(
            latent_mean.reshape(-1, self.latent_dim),
            latent_var.reshape(-1, self.latent_dim) ** 0.5
        )
        hi = sender_hidden.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(
            -1, self.args.rnn_hidden_dim
        )

        selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = F.one_hot(selected_action.squeeze(-1), num_classes=self.n_actions).float()
        one_hot_a = one_hot_a.view(sender_hidden.size(0), 1, self.n_agents, -1).expand(
            -1, self.n_agents, -1, -1
        )
        one_hot_a = one_hot_a.reshape(-1, self.n_actions)

        latent_infer = self.inference_net(th.cat([hi, one_hot_a], dim=-1)).view(
            sender_hidden.size(0) * self.n_agents * self.n_agents, -1
        )
        latent_infer[:, self.latent_dim:] = th.clamp(
            th.exp(latent_infer[:, self.latent_dim:]),
            min=self.args.var_floor
        )
        g2 = D.Normal(latent_infer[:, :self.latent_dim], latent_infer[:, self.latent_dim:] ** 0.5)
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = -(alpha * th.log2(alpha)).sum(dim=2).mean()
        return entropy_loss * self.args.entropy_loss_weight

    def build_logging_payload(self, alpha, head_messages):
        logs = {}
        detached_alpha = alpha.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-4) * th.log2(th.clamp(detached_alpha, min=1e-4))
        ).sum(dim=2).mean(dim=(0, 1))
        logs["Scalar_mean_attn_entropy"] = head_entropy.mean()

        for head_idx in range(self.attention_heads):
            logs["Scalar_head_{}_entropy".format(head_idx)] = head_entropy[head_idx]
            logs["Scalar_head_{}_feature_norm".format(head_idx)] = head_messages[:, :, head_idx, :].detach().norm(dim=-1).mean()
            if self.log_attention_maps:
                logs["Histogram_head_{}_attention".format(head_idx)] = detached_alpha[:, :, :, head_idx]
        return logs

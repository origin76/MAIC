import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions import kl_divergence


class MAICMultiHeadAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAICMultiHeadAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions
        self.attention_heads = max(1, getattr(args, "attention_heads", 1))
        self.attention_dim = args.attention_dim
        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        self.head_dim = self.attention_dim // self.attention_heads
        self.attn_diversity_loss_weight = getattr(args, "attn_diversity_loss_weight", 0.0)
        self.feat_diversity_loss_weight = getattr(args, "feat_diversity_loss_weight", 0.0)
        self.log_attention_maps = getattr(args, "log_attention_maps", False)

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

        self.msg_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.latent_dim, nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, self.attention_heads * args.n_actions)
        )

        self.w_query = nn.Linear(args.rnn_hidden_dim, self.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, self.attention_dim)
        self.head_fusion = nn.Identity() if self.attention_heads == 1 else nn.Linear(
            self.attention_heads * args.n_actions, args.n_actions
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
            gaussian_embed = D.Normal(latent_mean, latent_var ** 0.5)
            latent = gaussian_embed.rsample()

        sender_hidden = h.view(bs, self.n_agents, -1)
        repeated_sender_hidden = sender_hidden.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        msg_input = th.cat([repeated_sender_hidden, latent], dim=-1)
        msg = self.msg_net(msg_input.reshape(-1, msg_input.size(-1))).view(
            bs, self.n_agents, self.n_agents, self.attention_heads, self.n_actions
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

        gated_msg = alpha.unsqueeze(-1) * msg
        head_messages = th.sum(gated_msg, dim=1)
        comm_q = head_messages.reshape(bs * self.n_agents, self.attention_heads * self.n_actions)
        comm_q = self.head_fusion(comm_q)

        return_q = q + comm_q

        returns = {}
        if "train_mode" in kwargs and kwargs["train_mode"]:
            if hasattr(self.args, "mi_loss_weight") and self.args.mi_loss_weight > 0:
                returns["mi_loss"] = self.calculate_action_mi_loss(
                    sender_hidden, latent_mean, latent_var, return_q
                )
            if hasattr(self.args, "entropy_loss_weight") and self.args.entropy_loss_weight > 0:
                detached_query = self.w_query(sender_hidden.detach()).view(
                    bs, self.n_agents, self.attention_heads, self.head_dim
                )
                detached_key = self.w_key(latent.detach()).view(
                    bs, self.n_agents, self.n_agents, self.attention_heads, self.head_dim
                )
                detached_alpha = (detached_query.unsqueeze(2) * detached_key).sum(dim=-1)
                detached_alpha = detached_alpha.masked_fill(self_mask, -1e9)
                detached_alpha = F.softmax(detached_alpha / (self.head_dim ** 0.5), dim=2)
                returns["entropy_loss"] = self.calculate_entropy_loss(detached_alpha)
            if self.attn_diversity_loss_weight > 0:
                returns["attn_diversity_loss"] = self.calculate_attention_diversity_loss(alpha)
            if self.feat_diversity_loss_weight > 0:
                returns["feat_diversity_loss"] = self.calculate_feature_diversity_loss(head_messages)
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

    def calculate_attention_diversity_loss(self, alpha):
        if self.attention_heads <= 1 or self.n_agents <= 1:
            return alpha.new_zeros(())

        attn = alpha.permute(0, 1, 3, 2)
        attn = F.normalize(attn, p=2, dim=-1)
        gram = th.matmul(attn, attn.transpose(-1, -2))
        identity = th.eye(self.attention_heads, device=alpha.device).view(
            1, 1, self.attention_heads, self.attention_heads
        )
        diversity_loss = ((gram - identity) ** 2).mean()
        return diversity_loss * self.attn_diversity_loss_weight

    def calculate_feature_diversity_loss(self, head_messages):
        if self.attention_heads <= 1:
            return head_messages.new_zeros(())

        head_features = F.normalize(head_messages, p=2, dim=-1, eps=1e-8)
        cosine = th.matmul(head_features, head_features.transpose(-1, -2))
        identity = th.eye(self.attention_heads, device=head_messages.device).view(
            1, 1, self.attention_heads, self.attention_heads
        )
        off_diagonal = (cosine - identity) * (1 - identity)
        diversity_loss = off_diagonal.abs().sum() / (
            head_messages.size(0) * head_messages.size(1) * self.attention_heads * (self.attention_heads - 1)
        )
        return diversity_loss * self.feat_diversity_loss_weight

    def build_logging_payload(self, alpha, head_messages):
        logs = {}
        detached_alpha = alpha.detach()
        head_entropy = -(
            th.clamp(detached_alpha, min=1e-4) * th.log2(th.clamp(detached_alpha, min=1e-4))
        ).sum(dim=2).mean(dim=(0, 1))
        logs["Scalar_mean_attn_entropy"] = head_entropy.mean()
        if self.attention_heads > 1:
            attn = detached_alpha.permute(0, 1, 3, 2)
            attn = F.normalize(attn, p=2, dim=-1)
            gram = th.matmul(attn, attn.transpose(-1, -2))
            identity = th.eye(self.attention_heads, device=alpha.device).view(
                1, 1, self.attention_heads, self.attention_heads
            )
            logs["Scalar_attn_head_similarity"] = ((gram - identity) ** 2).mean()

            head_features = F.normalize(head_messages.detach(), p=2, dim=-1, eps=1e-8)
            cosine = th.matmul(head_features, head_features.transpose(-1, -2))
            off_diagonal = (cosine - identity) * (1 - identity)
            logs["Scalar_feat_head_similarity"] = off_diagonal.abs().mean()
        for head_idx in range(self.attention_heads):
            logs["Scalar_head_{}_entropy".format(head_idx)] = head_entropy[head_idx]
            if self.log_attention_maps:
                logs["Histogram_head_{}_attention".format(head_idx)] = detached_alpha[:, :, :, head_idx]
        return logs

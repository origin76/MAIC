import os

import torch as th
from torch.optim import Adam

from .budgeted_sparse_mappo_learner import BudgetedSparseMAPPOLearner
from modules.critics.mappo_agentwise_centralized import MAPPOAgentWiseCentralizedCritic
from utils.value_norm import ValueNorm


def huber_loss(error, delta):
    abs_error = error.abs()
    quadratic = th.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


class MAPPOAgentWiseCentralizedOfficialishLearner(BudgetedSparseMAPPOLearner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.actor_params = list(mac.parameters())
        self.critic = MAPPOAgentWiseCentralizedCritic(scheme, args)
        self.critic_params = list(self.critic.parameters())

        actor_optim_groups = self._build_actor_optim_groups()
        self.actor_optimiser = Adam(actor_optim_groups, lr=args.lr, eps=args.optim_eps)
        self.critic_optimiser = Adam(self.critic_params, lr=args.critic_lr, eps=args.optim_eps)
        self.initial_actor_lr = args.lr
        self.initial_critic_lr = args.critic_lr

        self.use_valuenorm = getattr(args, "use_valuenorm", True)
        self.use_huber_loss = getattr(args, "use_huber_loss", True)
        self.huber_delta = getattr(args, "huber_delta", 10.0)
        self.use_policy_active_masks = getattr(args, "use_policy_active_masks", True)
        self.use_value_active_masks = getattr(args, "use_value_active_masks", True)
        self.use_clipped_value_loss = getattr(args, "use_clipped_value_loss", True)
        self.use_linear_lr_decay = getattr(args, "use_linear_lr_decay", False)
        self.actor_min_lr_ratio = getattr(args, "actor_min_lr_ratio", getattr(args, "min_lr_ratio", 0.0))
        self.critic_min_lr_ratio = getattr(args, "critic_min_lr_ratio", getattr(args, "min_lr_ratio", 0.0))
        self.target_kl = getattr(args, "target_kl", None)

        self.value_normalizer = ValueNorm(1, device="cpu") if self.use_valuenorm else None
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _build_actor_optim_groups(self):
        if hasattr(self.mac.agent, "get_actor_optim_groups"):
            return self.mac.agent.get_actor_optim_groups(self.args.lr)
        return [{
            "params": self.actor_params,
            "lr": self.args.lr,
            "initial_lr": self.args.lr,
            "group_name": "actor",
        }]

    def train(self, batch, t_env: int, episode_num: int):
        if self.use_linear_lr_decay:
            self._update_learning_rate(t_env)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        active_masks = self._build_active_masks(batch)
        policy_mask = self._build_policy_mask(mask, active_masks)
        critic_mask = self._build_critic_mask(mask, active_masks)

        with th.no_grad():
            old_policy, _ = self._forward_policy(
                batch, prepare_for_logging=False, t_env=t_env
            )
            old_log_probs = self._get_action_log_probs(old_policy, actions)

            critic_outputs = self.critic(batch)
            values_for_returns = self._denormalize_values(critic_outputs)
            old_values = critic_outputs[:, :-1]
            returns, advantages = self._build_gae_targets(rewards, terminated, mask, values_for_returns)
            advantages = self._normalize_advantages(advantages, policy_mask)

            if self.value_normalizer is not None:
                self.value_normalizer.update(returns, mask=critic_mask)

        actor_log_stats = []
        critic_log_stats = []
        actor_logs = []
        value_clip_param = getattr(self.args, "value_clip_param", self.args.ppo_clip_param)
        kl_stop_triggered = False

        for epoch_idx in range(self.args.ppo_epochs):
            policy, extra = self._forward_policy(
                batch,
                prepare_for_logging=(epoch_idx == 0 and t_env - self.log_stats_t >= self.args.learner_log_interval),
                t_env=t_env,
            )
            new_log_probs = self._get_action_log_probs(policy, actions)
            entropy = self._policy_entropy(policy)

            ratio = th.exp(new_log_probs - old_log_probs)
            policy_advantages = advantages.squeeze(-1)

            surr1 = ratio * policy_advantages
            surr2 = th.clamp(
                ratio,
                1.0 - self.args.ppo_clip_param,
                1.0 + self.args.ppo_clip_param,
            ) * policy_advantages
            policy_loss = -(th.min(surr1, surr2) * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)
            entropy_loss = -(entropy * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)

            aux_loss, aux_loss_dict = self._process_extra_losses(extra, batch)
            actor_loss = policy_loss + self.args.entropy_coef * entropy_loss + aux_loss

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()

            values_pred = self.critic(batch)[:, :-1]
            clipped_values = old_values + (values_pred - old_values).clamp(
                min=-value_clip_param,
                max=value_clip_param,
            )

            value_target = self._normalize_returns(returns.detach())
            value_error = value_target - values_pred
            clipped_value_error = value_target - clipped_values

            if self.use_huber_loss:
                value_loss = huber_loss(value_error, self.huber_delta)
                clipped_value_loss = huber_loss(clipped_value_error, self.huber_delta)
            else:
                value_loss = value_error.pow(2)
                clipped_value_loss = clipped_value_error.pow(2)

            if self.use_clipped_value_loss:
                value_loss = th.max(value_loss, clipped_value_loss)

            value_loss = (value_loss * critic_mask).sum() / critic_mask.sum().clamp(min=1.0)

            self.critic_optimiser.zero_grad()
            (self.args.value_coef * value_loss).backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

            with th.no_grad():
                approx_kl = ((old_log_probs - new_log_probs) * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)
                clipfrac = (
                    (((ratio > (1.0 + self.args.ppo_clip_param)) | (ratio < (1.0 - self.args.ppo_clip_param))).float() * policy_mask).sum()
                    / policy_mask.sum().clamp(min=1.0)
                )
                value_mean = self._denormalize_values(values_pred).detach()

            actor_log_stats.append({
                "policy_loss": policy_loss.item(),
                "entropy": ((entropy * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)).item(),
                "actor_grad_norm": actor_grad_norm.item() if hasattr(actor_grad_norm, "item") else float(actor_grad_norm),
                "approx_kl": approx_kl.item(),
                "clipfrac": clipfrac.item(),
                "active_agent_ratio": (policy_mask.sum() / mask.expand(-1, -1, self.n_agents).sum().clamp(min=1.0)).item(),
                "actor_lr": self._get_actor_lr_for_group("actor_backbone", default_index=0),
            })
            if len(self.actor_optimiser.param_groups) > 1:
                actor_log_stats[-1]["comm_lr"] = self._get_actor_lr_for_group("actor_comm", default_index=1)
            critic_log_stats.append({
                "value_loss": value_loss.item(),
                "critic_grad_norm": critic_grad_norm.item() if hasattr(critic_grad_norm, "item") else float(critic_grad_norm),
                "critic_return_mean": ((returns * critic_mask).sum() / critic_mask.sum().clamp(min=1.0)).item(),
                "critic_value_mean": ((value_mean * critic_mask).sum() / critic_mask.sum().clamp(min=1.0)).item(),
                "critic_lr": self.critic_optimiser.param_groups[0]["lr"],
            })

            if extra.get("logs") is not None:
                actor_logs.extend(extra["logs"])
            for key, value in aux_loss_dict.items():
                actor_log_stats[-1][key] = value.item()

            if self.target_kl is not None and self.target_kl > 0 and approx_kl.item() > self.target_kl:
                kl_stop_triggered = True
                break

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self._log_epoch_stats(actor_log_stats, critic_log_stats, t_env)
            self._log_for_scalar_and_histogram(actor_logs, t_env)
            self.logger.log_stat("ppo_epochs_ran", float(len(actor_log_stats)), t_env)
            self.logger.log_stat("kl_early_stop", 1.0 if kl_stop_triggered else 0.0, t_env)
            self.log_stats_t = t_env

    def _build_active_masks(self, batch):
        avail_actions = batch["avail_actions"][:, :-1]
        active = (avail_actions.sum(dim=-1, keepdim=True) > 1).float()
        return active

    def _build_policy_mask(self, mask, active_masks):
        policy_mask = mask.expand(-1, -1, self.n_agents)
        if self.use_policy_active_masks:
            policy_mask = policy_mask * active_masks.squeeze(-1)
        return policy_mask

    def _build_critic_mask(self, mask, active_masks):
        critic_mask = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        if self.use_value_active_masks:
            critic_mask = critic_mask * active_masks
        return critic_mask

    def _normalize_returns(self, returns):
        if self.value_normalizer is None:
            return returns
        return self.value_normalizer.normalize(returns)

    def _denormalize_values(self, values):
        if self.value_normalizer is None:
            return values
        return self.value_normalizer.denormalize(values)

    def _build_gae_targets(self, rewards, terminated, mask, values):
        values = values.squeeze(-1)
        rewards = rewards.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.n_agents)
        terminated = terminated.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.n_agents)
        mask = mask.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.n_agents)

        advantages = th.zeros_like(rewards)
        gae = th.zeros_like(rewards[:, 0])

        for t in reversed(range(rewards.size(1))):
            next_value = values[:, t + 1]
            delta = rewards[:, t] + self.args.gamma * next_value * (1 - terminated[:, t]) - values[:, t]
            gae = delta + self.args.gamma * self.args.gae_lambda * (1 - terminated[:, t]) * gae
            gae = gae * mask[:, t]
            advantages[:, t] = gae

        returns = advantages + values[:, :-1]
        return returns.unsqueeze(-1), advantages.unsqueeze(-1)

    def _normalize_advantages(self, advantages, policy_mask):
        advantage_mask = policy_mask.unsqueeze(-1)
        valid_advantages = advantages[advantage_mask.bool()]
        if valid_advantages.numel() == 0:
            return advantages
        normalized = (advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-6)
        return normalized

    def _update_learning_rate(self, t_env):
        progress = min(1.0, max(0.0, float(t_env) / float(max(1, self.args.t_max))))
        actor_decay = 1.0 - (1.0 - self.actor_min_lr_ratio) * progress
        critic_decay = 1.0 - (1.0 - self.critic_min_lr_ratio) * progress

        critic_lr = self.initial_critic_lr * critic_decay

        for param_group in self.actor_optimiser.param_groups:
            initial_lr = param_group.get("initial_lr", self.initial_actor_lr)
            param_group["lr"] = initial_lr * actor_decay
        for param_group in self.critic_optimiser.param_groups:
            param_group["lr"] = critic_lr

    def _get_actor_lr_for_group(self, group_name, default_index=0):
        for idx, param_group in enumerate(self.actor_optimiser.param_groups):
            if param_group.get("group_name") == group_name:
                return param_group["lr"]
        fallback_index = min(default_index, len(self.actor_optimiser.param_groups) - 1)
        return self.actor_optimiser.param_groups[fallback_index]["lr"]

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        if self.value_normalizer is not None:
            self.value_normalizer.to("cuda")

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        if self.value_normalizer is not None:
            th.save(self.value_normalizer.state_dict(), "{}/value_norm.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.actor_optimiser.load_state_dict(th.load("{}/actor_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.value_normalizer is not None:
            self.value_normalizer.load_state_dict(
                th.load("{}/value_norm.th".format(path), map_location=lambda storage, loc: storage)
            )

    def init_models(self, path, strict=False, load_actor=True, load_critic=True, load_value_norm=True, **kwargs):
        report = super().init_models(path, strict=strict, load_actor=load_actor, load_critic=load_critic)

        if load_value_norm and self.value_normalizer is not None:
            value_norm_path = "{}/value_norm.th".format(path)
            if os.path.exists(value_norm_path):
                value_norm_state = th.load(value_norm_path, map_location=lambda storage, loc: storage)
                self.value_normalizer.load_state_dict(value_norm_state)
                report["value_norm"] = "loaded"
            else:
                report["value_norm"] = "missing"

        return report

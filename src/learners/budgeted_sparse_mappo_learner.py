import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from modules.critics.mappo import MAPPOCritic


class BudgetedSparseMAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.actor_params = list(mac.parameters())
        self.critic = MAPPOCritic(scheme, args)
        self.critic_params = list(self.critic.parameters())

        self.actor_optimiser = Adam(self.actor_params, lr=args.lr, eps=args.optim_eps)
        self.critic_optimiser = Adam(self.critic_params, lr=args.critic_lr, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        with th.no_grad():
            old_policy, _ = self._forward_policy(batch, prepare_for_logging=False)
            old_log_probs = self._get_action_log_probs(old_policy, actions)

            values = self.critic(batch)
            returns, advantages = self._build_gae_targets(rewards, terminated, mask, values)
            advantages = self._normalize_advantages(advantages, mask)

        actor_log_stats = []
        critic_log_stats = []
        actor_logs = []

        for epoch_idx in range(self.args.ppo_epochs):
            policy, extra = self._forward_policy(
                batch,
                prepare_for_logging=(epoch_idx == 0 and t_env - self.log_stats_t >= self.args.learner_log_interval),
            )
            new_log_probs = self._get_action_log_probs(policy, actions)
            entropy = self._policy_entropy(policy)

            ratio = th.exp(new_log_probs - old_log_probs)
            expanded_advantages = advantages.expand(-1, -1, self.n_agents)
            expanded_mask = mask.expand(-1, -1, self.n_agents)

            surr1 = ratio * expanded_advantages
            surr2 = th.clamp(
                ratio,
                1.0 - self.args.ppo_clip_param,
                1.0 + self.args.ppo_clip_param,
            ) * expanded_advantages
            policy_loss = - (th.min(surr1, surr2) * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)
            entropy_loss = - (entropy * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)

            aux_loss, aux_loss_dict = self._process_extra_losses(extra, batch)
            actor_loss = policy_loss + self.args.entropy_coef * entropy_loss + aux_loss

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()

            values_pred = self.critic(batch)[:, :-1]
            value_loss = (((values_pred - returns.detach()) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

            self.critic_optimiser.zero_grad()
            (self.args.value_coef * value_loss).backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

            with th.no_grad():
                approx_kl = (((old_log_probs - new_log_probs) * expanded_mask).sum() /
                             expanded_mask.sum().clamp(min=1.0))
                clipfrac = ((((ratio > (1.0 + self.args.ppo_clip_param)) |
                              (ratio < (1.0 - self.args.ppo_clip_param))).float() * expanded_mask).sum() /
                            expanded_mask.sum().clamp(min=1.0))

            actor_log_stats.append({
                "policy_loss": policy_loss.item(),
                "entropy": ((entropy * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)).item(),
                "actor_grad_norm": actor_grad_norm.item() if hasattr(actor_grad_norm, "item") else float(actor_grad_norm),
                "approx_kl": approx_kl.item(),
                "clipfrac": clipfrac.item(),
            })
            critic_log_stats.append({
                "value_loss": value_loss.item(),
                "critic_grad_norm": critic_grad_norm.item() if hasattr(critic_grad_norm, "item") else float(critic_grad_norm),
                "critic_return_mean": ((returns * mask).sum() / mask.sum().clamp(min=1.0)).item(),
                "critic_value_mean": ((values_pred.detach() * mask).sum() / mask.sum().clamp(min=1.0)).item(),
            })

            if extra.get("logs") is not None:
                actor_logs.extend(extra["logs"])
            for key, value in aux_loss_dict.items():
                actor_log_stats[-1][key] = value.item()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self._log_epoch_stats(actor_log_stats, critic_log_stats, t_env)
            self._log_for_scalar_and_histogram(actor_logs, t_env)
            self.log_stats_t = t_env

    def _forward_policy(self, batch, prepare_for_logging=False):
        outputs = []
        loss_items = []
        logs = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, extra = self.mac.forward(
                batch,
                t=t,
                test_mode=False,
                train_mode=True,
                prepare_for_logging=prepare_for_logging,
            )
            outputs.append(agent_outs)
            if "logs" in extra:
                logs.append(extra["logs"])
                del extra["logs"]
            loss_items.append(extra)
        policy = th.stack(outputs, dim=1)
        merged = self._merge_extra_items(loss_items, logs)
        return policy, merged

    def _merge_extra_items(self, loss_items, logs):
        merged = {}
        for item in loss_items:
            for key, value in item.items():
                merged[key] = merged.get(key, 0) + value
        for key in list(merged.keys()):
            merged[key] = merged[key] / max(1, len(loss_items))
        if len(logs) > 0:
            merged["logs"] = logs
        return merged

    def _process_extra_losses(self, extra, batch):
        total = 0
        loss_dict = {}
        for key, value in extra.items():
            if str(key).endswith("loss"):
                total = total + value
                loss_dict[key] = value.detach()
        if not loss_dict:
            total = batch["reward"].new_zeros(())
        return total, loss_dict

    def _get_action_log_probs(self, policy, actions):
        chosen_probs = th.gather(policy, dim=3, index=actions).squeeze(3)
        return th.log(chosen_probs.clamp(min=1e-10))

    def _policy_entropy(self, policy):
        return -(policy.clamp(min=1e-10) * th.log(policy.clamp(min=1e-10))).sum(dim=3)

    def _build_gae_targets(self, rewards, terminated, mask, values):
        values = values.squeeze(-1)
        rewards = rewards.squeeze(-1)
        terminated = terminated.squeeze(-1)
        mask = mask.squeeze(-1)

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

    def _normalize_advantages(self, advantages, mask):
        valid_advantages = advantages[mask.bool()]
        if valid_advantages.numel() == 0:
            return advantages
        normalized = (advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-6)
        return normalized

    def _log_epoch_stats(self, actor_stats, critic_stats, t_env):
        actor_keys = actor_stats[0].keys()
        critic_keys = critic_stats[0].keys()
        for key in actor_keys:
            self.logger.log_stat(key, sum(item[key] for item in actor_stats) / len(actor_stats), t_env)
        for key in critic_keys:
            self.logger.log_stat(key, sum(item[key] for item in critic_stats) / len(critic_stats), t_env)

    def _log_for_scalar_and_histogram(self, logs, t):
        if len(logs) == 0:
            return
        keys = list(logs[0].keys())
        for key in keys:
            log_key = "_".join(key.split("_")[1:])
            if str(key).startswith("Histogram"):
                if not getattr(self.logger, "use_tb", False):
                    continue
                value = th.stack([item[key] for item in logs], dim=1)
                self.logger.log_histogram(log_key, value, t)
            elif str(key).startswith("Scalar"):
                values = [item[key].float() if isinstance(item[key], th.Tensor) else th.tensor(item[key]) for item in logs]
                self.logger.log_stat(log_key, th.stack(values).mean(), t)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.actor_optimiser.load_state_dict(th.load("{}/actor_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

import torch as th

from .budgeted_sparse_mappo_learner import BudgetedSparseMAPPOLearner


class BudgetedSparseMAPPOVClipLearner(BudgetedSparseMAPPOLearner):
    def train(self, batch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        with th.no_grad():
            old_policy, _ = self._forward_policy(batch, prepare_for_logging=False)
            old_log_probs = self._get_action_log_probs(old_policy, actions)

            values = self.critic(batch)
            old_values = values[:, :-1]
            returns, advantages = self._build_gae_targets(rewards, terminated, mask, values)
            advantages = self._normalize_advantages(advantages, mask)

        actor_log_stats = []
        critic_log_stats = []
        actor_logs = []
        value_clip_param = getattr(self.args, "value_clip_param", self.args.ppo_clip_param)

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
            clipped_values = old_values + (values_pred - old_values).clamp(
                min=-value_clip_param,
                max=value_clip_param,
            )
            value_error = (values_pred - returns.detach()) ** 2
            clipped_value_error = (clipped_values - returns.detach()) ** 2
            value_loss = (th.max(value_error, clipped_value_error) * mask).sum() / mask.sum().clamp(min=1.0)

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

import os

import torch as th
import torch.nn.functional as F
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

        self.actor_weight_decay = float(getattr(args, "actor_weight_decay", 0.0))
        self.critic_weight_decay = float(getattr(args, "critic_weight_decay", 0.0))
        actor_optim_groups = self._build_actor_optim_groups()
        self.actor_optimiser = Adam(
            actor_optim_groups,
            lr=args.lr,
            eps=args.optim_eps,
            weight_decay=self.actor_weight_decay,
        )
        self.critic_optimiser = Adam(
            self.critic_params,
            lr=args.critic_lr,
            eps=args.optim_eps,
            weight_decay=self.critic_weight_decay,
        )
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
        self.advantage_clip = getattr(args, "advantage_clip", None)
        self.counterfactual_usegate_soft_weighting = bool(
            getattr(args, "counterfactual_usegate_soft_weighting", False)
        )
        self.counterfactual_usegate_weight_temp = float(
            getattr(
                args,
                "counterfactual_usegate_weight_temp",
                getattr(args, "counterfactual_usegate_gain_temp", 0.1),
            )
        )
        self.counterfactual_usegate_soft_weight_fixed_denom = bool(
            getattr(args, "counterfactual_usegate_soft_weight_fixed_denom", False)
        )
        # Overpredict loss is a "close-gate" pressure. By default it applies on all
        # samples, which can become too harsh when gains shrink late in training.
        # These switches let us apply it only when gains are (confidently) negative.
        self.counterfactual_usegate_overpredict_only_negative = bool(
            getattr(args, "counterfactual_usegate_overpredict_only_negative", False)
        )
        self.counterfactual_usegate_overpredict_soft_weighting = bool(
            getattr(args, "counterfactual_usegate_overpredict_soft_weighting", False)
        )
        self.counterfactual_usegate_gain_norm = bool(
            getattr(args, "counterfactual_usegate_gain_norm", False)
        )
        self.counterfactual_usegate_gain_norm_ema_decay = float(
            getattr(args, "counterfactual_usegate_gain_norm_ema_decay", 0.99)
        )
        self.counterfactual_usegate_gain_norm_eps = float(
            getattr(args, "counterfactual_usegate_gain_norm_eps", 1e-6)
        )
        self.counterfactual_usegate_attack_gain_norm_eps = float(
            getattr(
                args,
                "counterfactual_usegate_attack_gain_norm_eps",
                self.counterfactual_usegate_gain_norm_eps,
            )
        )
        self.counterfactual_usegate_move_gain_norm_eps = float(
            getattr(
                args,
                "counterfactual_usegate_move_gain_norm_eps",
                self.counterfactual_usegate_gain_norm_eps,
            )
        )
        self.attack_adv_leverage_loss_weight = float(
            getattr(args, "attack_adv_leverage_loss_weight", 0.0)
        )
        self.attack_adv_leverage_margin = float(
            getattr(args, "attack_adv_leverage_margin", 0.03)
        )
        self.attack_adv_leverage_adv_clip = getattr(
            args, "attack_adv_leverage_adv_clip", 2.0
        )
        self.attack_adv_leverage_use_real_comm_weight = bool(
            getattr(args, "attack_adv_leverage_use_real_comm_weight", True)
        )
        self.attack_adv_leverage_fixed_denom = bool(
            getattr(args, "attack_adv_leverage_fixed_denom", True)
        )
        self.attack_adv_leverage_loss_mode = str(
            getattr(args, "attack_adv_leverage_loss_mode", "squared")
        ).lower()
        self.attack_adv_leverage_huber_beta = float(
            getattr(args, "attack_adv_leverage_huber_beta", 0.03)
        )
        if self.attack_adv_leverage_loss_mode not in {"squared", "huber", "linear"}:
            raise ValueError(
                "attack_adv_leverage_loss_mode must be one of: squared, huber, linear"
            )
        if self.attack_adv_leverage_huber_beta <= 0:
            raise ValueError("attack_adv_leverage_huber_beta must be positive")
        self.attack_causal_leverage_loss_weight = float(
            getattr(args, "attack_causal_leverage_loss_weight", 0.0)
        )
        self.attack_causal_leverage_margin = float(
            getattr(args, "attack_causal_leverage_margin", 0.02)
        )
        self.attack_causal_leverage_huber_beta = float(
            getattr(args, "attack_causal_leverage_huber_beta", 0.02)
        )
        self.attack_causal_leverage_adv_clip = getattr(
            args, "attack_causal_leverage_adv_clip", 2.0
        )
        self.attack_causal_leverage_use_real_comm_weight = bool(
            getattr(args, "attack_causal_leverage_use_real_comm_weight", True)
        )
        self.attack_causal_leverage_fixed_denom = bool(
            getattr(args, "attack_causal_leverage_fixed_denom", True)
        )
        self.attack_causal_local_target_prob_threshold = float(
            getattr(args, "attack_causal_local_target_prob_threshold", 0.35)
        )
        self.attack_causal_peer_support_threshold = float(
            getattr(args, "attack_causal_peer_support_threshold", 0.25)
        )
        self.attack_causal_stability_loss_weight = float(
            getattr(args, "attack_causal_stability_loss_weight", 0.0)
        )
        self.attack_causal_bad_loss_weight = float(
            getattr(args, "attack_causal_bad_loss_weight", 0.0)
        )
        self.attack_causal_bad_margin = float(
            getattr(args, "attack_causal_bad_margin", 0.02)
        )
        self.attack_causal_bad_huber_beta = float(
            getattr(args, "attack_causal_bad_huber_beta", 0.02)
        )
        self.attack_causal_bad_adv_clip = getattr(
            args, "attack_causal_bad_adv_clip", 2.0
        )
        self.attack_causal_bad_peer_support_threshold = float(
            getattr(args, "attack_causal_bad_peer_support_threshold", 0.55)
        )
        self.attack_causal_bad_peer_boost_weight = float(
            getattr(args, "attack_causal_bad_peer_boost_weight", 0.5)
        )
        self.attack_peer_conflict_margin_leverage_loss_weight = float(
            getattr(args, "attack_peer_conflict_margin_leverage_loss_weight", 0.0)
        )
        self.attack_peer_conflict_attack_only_margin_loss_weight = float(
            getattr(args, "attack_peer_conflict_attack_only_margin_loss_weight", 0.0)
        )
        self.attack_peer_conflict_margin = float(
            getattr(args, "attack_peer_conflict_margin", 0.03)
        )
        self.attack_peer_conflict_margin_huber_beta = float(
            getattr(args, "attack_peer_conflict_margin_huber_beta", 0.02)
        )
        self.attack_peer_conflict_peer_support_threshold = float(
            getattr(
                args, "attack_peer_conflict_peer_support_threshold", 0.7
            )
        )
        self.attack_peer_conflict_use_real_comm_weight = bool(
            getattr(args, "attack_peer_conflict_use_real_comm_weight", True)
        )
        self.attack_peer_conflict_fixed_denom = bool(
            getattr(args, "attack_peer_conflict_fixed_denom", True)
        )
        fused_local_conf_max = getattr(
            args, "attack_peer_conflict_fused_local_conf_max", None
        )
        self.attack_peer_conflict_fused_local_conf_max = (
            None if fused_local_conf_max is None else float(fused_local_conf_max)
        )
        self.attack_peer_conflict_fused_local_uncertainty_min_weight = float(
            getattr(
                args,
                "attack_peer_conflict_fused_local_uncertainty_min_weight",
                0.2,
            )
        )
        if self.attack_causal_leverage_huber_beta <= 0:
            raise ValueError("attack_causal_leverage_huber_beta must be positive")
        if self.attack_causal_bad_huber_beta <= 0:
            raise ValueError("attack_causal_bad_huber_beta must be positive")
        if self.attack_causal_local_target_prob_threshold < 0:
            raise ValueError(
                "attack_causal_local_target_prob_threshold must be non-negative"
            )
        if self.attack_causal_peer_support_threshold < 0:
            raise ValueError(
                "attack_causal_peer_support_threshold must be non-negative"
            )
        if self.attack_causal_bad_peer_support_threshold < 0:
            raise ValueError(
                "attack_causal_bad_peer_support_threshold must be non-negative"
            )
        if self.attack_causal_bad_peer_boost_weight < 0:
            raise ValueError(
                "attack_causal_bad_peer_boost_weight must be non-negative"
            )
        if self.attack_peer_conflict_margin_leverage_loss_weight < 0:
            raise ValueError(
                "attack_peer_conflict_margin_leverage_loss_weight must be non-negative"
            )
        if self.attack_peer_conflict_attack_only_margin_loss_weight < 0:
            raise ValueError(
                "attack_peer_conflict_attack_only_margin_loss_weight must be non-negative"
            )
        if self.attack_peer_conflict_margin <= 0:
            raise ValueError("attack_peer_conflict_margin must be positive")
        if self.attack_peer_conflict_margin_huber_beta <= 0:
            raise ValueError("attack_peer_conflict_margin_huber_beta must be positive")
        if self.attack_peer_conflict_peer_support_threshold < 0 or self.attack_peer_conflict_peer_support_threshold > 1:
            raise ValueError(
                "attack_peer_conflict_peer_support_threshold must be in [0, 1]"
            )
        if (
            self.attack_peer_conflict_fused_local_conf_max is not None
            and (
                self.attack_peer_conflict_fused_local_conf_max <= 0
                or self.attack_peer_conflict_fused_local_conf_max > 1
            )
        ):
            raise ValueError(
                "attack_peer_conflict_fused_local_conf_max must be in (0, 1]"
            )
        if (
            self.attack_peer_conflict_fused_local_uncertainty_min_weight < 0
            or self.attack_peer_conflict_fused_local_uncertainty_min_weight > 1
        ):
            raise ValueError(
                "attack_peer_conflict_fused_local_uncertainty_min_weight must be in [0, 1]"
            )
        self._attack_gain_abs_ema = None
        self._move_gain_abs_ema = None

        self.comm_warmup_steps = int(getattr(args, "comm_warmup_steps", 0))
        self.comm_warmup_delay_steps = int(getattr(args, "comm_warmup_delay_steps", 0))
        self.comm_warmup_start_factor = float(getattr(args, "comm_warmup_start_factor", 1.0))
        self.comm_warmup_end_factor = float(getattr(args, "comm_warmup_end_factor", 1.0))
        self.comm_warmup_exponent = float(getattr(args, "comm_warmup_exponent", 1.0))
        self._t_env = 0

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
        self._t_env = t_env
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
                batch,
                prepare_for_logging=False,
                t_env=t_env,
                collect_sequence_data=False,
            )
            old_log_probs = self._get_action_log_probs(old_policy, actions)

            critic_outputs = self.critic(batch)
            values_for_returns = self._denormalize_values(critic_outputs)
            old_values = critic_outputs[:, :-1]
            returns, advantages = self._build_gae_targets(rewards, terminated, mask, values_for_returns)
            advantages = self._normalize_advantages(advantages, policy_mask)
            if self.advantage_clip is not None:
                clip_value = float(self.advantage_clip)
                if clip_value > 0:
                    advantages = advantages.clamp(min=-clip_value, max=clip_value)

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
                collect_sequence_data=(
                    getattr(self.args, "counterfactual_usegate", False)
                    or self.attack_adv_leverage_loss_weight > 0
                    or self.attack_causal_leverage_loss_weight > 0
                    or self.attack_causal_stability_loss_weight > 0
                    or self.attack_causal_bad_loss_weight > 0
                    or self.attack_peer_conflict_margin_leverage_loss_weight > 0
                    or self.attack_peer_conflict_attack_only_margin_loss_weight > 0
                ),
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

            aux_loss, aux_loss_dict = self._process_extra_losses(
                extra,
                batch,
                actions=actions,
                advantages=policy_advantages,
                policy_mask=policy_mask,
            )
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

    def _forward_policy(
        self,
        batch,
        prepare_for_logging=False,
        t_env=None,
        collect_sequence_data=False,
    ):
        outputs = []
        loss_items = []
        logs = []
        sequence_items = {}

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, extra = self.mac.forward(
                batch,
                t=t,
                test_mode=False,
                train_mode=True,
                prepare_for_logging=prepare_for_logging,
                t_env=t_env,
                collect_sequence_data=collect_sequence_data,
                chosen_actions=batch["actions"][:, t],
            )
            outputs.append(agent_outs)

            if "logs" in extra:
                logs.append(extra["logs"])
                del extra["logs"]

            for key in list(extra.keys()):
                if str(key).startswith("seq_"):
                    sequence_items.setdefault(key, []).append(extra.pop(key))

            loss_items.append(extra)

        policy = th.stack(outputs, dim=1)
        merged = self._merge_extra_items(loss_items, logs)
        for key, values in sequence_items.items():
            merged[key] = th.stack(values, dim=1)
        return policy, merged

    def _process_extra_losses(
        self,
        extra,
        batch,
        actions=None,
        advantages=None,
        policy_mask=None,
    ):
        total, loss_dict = super()._process_extra_losses(extra, batch)

        total, loss_dict = self._process_attack_adv_leverage_loss(
            total,
            loss_dict,
            extra,
            batch,
            actions=actions,
            advantages=advantages,
            policy_mask=policy_mask,
        )
        total, loss_dict = self._process_attack_causal_leverage_loss(
            total,
            loss_dict,
            extra,
            batch,
            actions=actions,
            advantages=advantages,
            policy_mask=policy_mask,
        )
        total, loss_dict = self._process_attack_peer_conflict_margin_leverage_loss(
            total,
            loss_dict,
            extra,
            batch,
            actions=actions,
            policy_mask=policy_mask,
        )

        if (
            not getattr(self.args, "counterfactual_usegate", False)
            or actions is None
            or advantages is None
            or policy_mask is None
        ):
            return total, loss_dict

        usegate_pred_keys = (
            "seq_counterfactual_attack_usegate_pred",
            "seq_counterfactual_move_usegate_pred",
        )
        if any(key not in extra for key in usegate_pred_keys):
            return total, loss_dict

        logp_keys = (
            "seq_counterfactual_local_logp",
            "seq_counterfactual_attack_logp",
            "seq_counterfactual_move_logp",
        )
        policy_keys = (
            "seq_counterfactual_local_policy",
            "seq_counterfactual_attack_policy",
            "seq_counterfactual_move_policy",
        )
        has_logp = all(key in extra for key in logp_keys)
        has_policy = all(key in extra for key in policy_keys)
        if not has_logp and not has_policy:
            return total, loss_dict

        attack_pred = extra["seq_counterfactual_attack_usegate_pred"].squeeze(-1)
        move_pred = extra["seq_counterfactual_move_usegate_pred"].squeeze(-1)

        if has_logp:
            local_log_probs = extra["seq_counterfactual_local_logp"]
            fused_log_probs = extra.get("seq_counterfactual_fused_logp", None)
            attack_log_probs = extra["seq_counterfactual_attack_logp"]
            move_log_probs = extra["seq_counterfactual_move_logp"]
        else:
            fused_log_probs = None
            local_policy = extra["seq_counterfactual_local_policy"]
            attack_policy = extra["seq_counterfactual_attack_policy"]
            move_policy = extra["seq_counterfactual_move_policy"]
            local_log_probs = self._get_action_log_probs(local_policy, actions)
            attack_log_probs = self._get_action_log_probs(attack_policy, actions)
            move_log_probs = self._get_action_log_probs(move_policy, actions)

        action_ids = actions.squeeze(-1)
        semantic_action_offset = int(
            getattr(self.args, "semantic_action_offset", 6)
        )
        attack_mask = (
            (action_ids >= semantic_action_offset).float() * policy_mask
        )
        move_mask = (
            (action_ids < semantic_action_offset).float() * policy_mask
        )

        attack_gain = advantages * (attack_log_probs - local_log_probs)
        move_gain = advantages * (move_log_probs - local_log_probs)
        attack_gain_for_target = attack_gain
        move_gain_for_target = move_gain
        attack_gain_scale = attack_gain.new_tensor(1.0)
        move_gain_scale = move_gain.new_tensor(1.0)

        if self.counterfactual_usegate_gain_norm:
            attack_gain_for_target, attack_gain_scale = self._normalize_usegate_gain(
                attack_gain, attack_mask, stream_name="attack"
            )
            move_gain_for_target, move_gain_scale = self._normalize_usegate_gain(
                move_gain, move_mask, stream_name="move"
            )

        gain_temp = max(
            1e-6,
            float(getattr(self.args, "counterfactual_usegate_gain_temp", 0.1)),
        )
        attack_target = (
            1.0 - th.exp(-F.relu(attack_gain_for_target.detach()) / gain_temp)
        )
        move_target = (
            1.0 - th.exp(-F.relu(move_gain_for_target.detach()) / gain_temp)
        )

        zero = batch["reward"].new_zeros(())
        attack_loss = zero
        move_loss = zero
        sparsity_loss = zero

        attack_denom = attack_mask.sum().clamp(min=1.0)
        move_denom = move_mask.sum().clamp(min=1.0)
        full_denom = policy_mask.sum().clamp(min=1.0)
        step_mask = (policy_mask.sum(dim=-1) > 0).float()
        step_denom = step_mask.sum().clamp(min=1.0)

        # Usegate loss is applied only where gain > 0 (push gate open).
        # Sparsity handles the default-closed direction; applying loss toward
        # target=0 when gain<=0 would lock the gate shut against noisy baselines.
        weight_temp = max(1e-6, self.counterfactual_usegate_weight_temp)
        if self.counterfactual_usegate_soft_weighting:
            # Zero-baseline soft weighting:
            #   old: sigmoid(x)        -> weight(0)=0.5
            #   new: relu(2*sigmoid(x)-1) -> weight(0)=0
            # This avoids a constant open-gate pressure when gain is near zero.
            positive_attack_mask = (
                F.relu(
                    2.0 * th.sigmoid(attack_gain_for_target.detach() / weight_temp) - 1.0
                )
                * attack_mask
            )
            positive_move_mask = (
                F.relu(
                    2.0 * th.sigmoid(move_gain_for_target.detach() / weight_temp) - 1.0
                )
                * move_mask
            )
        else:
            positive_attack_mask = (attack_gain_for_target.detach() > 0).float() * attack_mask
            positive_move_mask = (move_gain_for_target.detach() > 0).float() * move_mask

        if self.counterfactual_usegate_overpredict_soft_weighting:
            negative_attack_mask = (
                F.relu(
                    2.0 * th.sigmoid(-attack_gain_for_target.detach() / weight_temp) - 1.0
                )
                * attack_mask
            )
            negative_move_mask = (
                F.relu(
                    2.0 * th.sigmoid(-move_gain_for_target.detach() / weight_temp) - 1.0
                )
                * move_mask
            )
        else:
            negative_attack_mask = (attack_gain_for_target.detach() < 0).float() * attack_mask
            negative_move_mask = (move_gain_for_target.detach() < 0).float() * move_mask

        attack_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_attack_loss_weight",
                0.0,
            )
        )
        move_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_move_loss_weight",
                0.0,
            )
        )
        attack_sparse_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_attack_sparsity_weight",
                0.0,
            )
        )
        move_sparse_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_move_sparsity_weight",
                0.0,
            )
        )
        attack_overpredict_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_attack_overpredict_weight",
                0.0,
            )
        )
        move_overpredict_weight = float(
            getattr(
                self.args,
                "counterfactual_usegate_move_overpredict_weight",
                0.0,
            )
        )

        warmup_scale = self._compute_comm_warmup_factor()
        if warmup_scale < 1.0:
            attack_weight *= warmup_scale
            move_weight *= warmup_scale
            # sparsity and overpredict NOT scaled: closing pressure active from day 1

        if attack_weight > 0:
            attack_loss_denom = positive_attack_mask.sum().clamp(min=1.0)
            if (
                self.counterfactual_usegate_soft_weighting
                and self.counterfactual_usegate_soft_weight_fixed_denom
            ):
                attack_loss_denom = attack_denom
            attack_loss = attack_weight * (
                ((attack_pred - attack_target).pow(2) * positive_attack_mask).sum()
                / attack_loss_denom
            )
        if move_weight > 0:
            move_loss_denom = positive_move_mask.sum().clamp(min=1.0)
            if (
                self.counterfactual_usegate_soft_weighting
                and self.counterfactual_usegate_soft_weight_fixed_denom
            ):
                move_loss_denom = move_denom
            move_loss = move_weight * (
                ((move_pred - move_target).pow(2) * positive_move_mask).sum()
                / move_loss_denom
            )
        if attack_sparse_weight > 0:
            sparsity_loss = sparsity_loss + attack_sparse_weight * (
                (attack_pred * policy_mask).sum() / full_denom
            )
        if move_sparse_weight > 0:
            sparsity_loss = sparsity_loss + move_sparse_weight * (
                (move_pred * policy_mask).sum() / full_denom
            )

        attack_overpredict_loss = zero
        move_overpredict_loss = zero
        if attack_overpredict_weight > 0:
            attack_overpredict_mask = (
                negative_attack_mask
                if self.counterfactual_usegate_overpredict_only_negative
                else attack_mask
            )
            attack_overpredict_loss = attack_overpredict_weight * (
                (F.relu(attack_pred - attack_target).pow(2) * attack_overpredict_mask).sum()
                / attack_denom
            )
        if move_overpredict_weight > 0:
            move_overpredict_mask = (
                negative_move_mask
                if self.counterfactual_usegate_overpredict_only_negative
                else move_mask
            )
            move_overpredict_loss = move_overpredict_weight * (
                (F.relu(move_pred - move_target).pow(2) * move_overpredict_mask).sum()
                / move_denom
            )

        total = (
            total
            + attack_loss
            + move_loss
            + sparsity_loss
            + attack_overpredict_loss
            + move_overpredict_loss
        )

        loss_dict["counterfactual_attack_usegate_loss"] = attack_loss.detach()
        loss_dict["counterfactual_move_usegate_loss"] = move_loss.detach()
        if attack_overpredict_weight > 0:
            loss_dict["counterfactual_attack_overpredict_loss"] = attack_overpredict_loss.detach()
        if move_overpredict_weight > 0:
            loss_dict["counterfactual_move_overpredict_loss"] = move_overpredict_loss.detach()
        if attack_sparse_weight > 0 or move_sparse_weight > 0:
            loss_dict["counterfactual_usegate_sparsity_loss"] = sparsity_loss.detach()
        loss_dict["counterfactual_attack_gain_mean"] = (
            (attack_gain.detach() * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_move_gain_mean"] = (
            (move_gain.detach() * move_mask).sum() / move_denom
        )
        loss_dict["counterfactual_attack_target_mean"] = (
            (attack_target * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_move_target_mean"] = (
            (move_target * move_mask).sum() / move_denom
        )
        loss_dict["counterfactual_attack_pred_mean"] = (
            (attack_pred.detach() * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_move_pred_mean"] = (
            (move_pred.detach() * move_mask).sum() / move_denom
        )
        loss_dict["counterfactual_attack_positive_ratio"] = (
            ((attack_gain.detach() > 0).float() * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_move_positive_ratio"] = (
            ((move_gain.detach() > 0).float() * move_mask).sum() / move_denom
        )
        loss_dict["counterfactual_attack_gain_scale_ema"] = attack_gain_scale.detach()
        loss_dict["counterfactual_move_gain_scale_ema"] = move_gain_scale.detach()
        if self.counterfactual_usegate_soft_weighting:
            loss_dict["counterfactual_attack_soft_weight_mean"] = (
                positive_attack_mask.sum() / attack_denom
            )
            loss_dict["counterfactual_move_soft_weight_mean"] = (
                positive_move_mask.sum() / move_denom
            )
        loss_dict["counterfactual_attack_overpredict_mean"] = (
            (F.relu(attack_pred.detach() - attack_target) * attack_mask).sum()
            / attack_denom
        )
        loss_dict["counterfactual_move_overpredict_mean"] = (
            (F.relu(move_pred.detach() - move_target) * move_mask).sum()
            / move_denom
        )
        if self.counterfactual_usegate_overpredict_soft_weighting:
            loss_dict["counterfactual_attack_negative_soft_weight_mean"] = (
                negative_attack_mask.sum() / attack_denom
            )
            loss_dict["counterfactual_move_negative_soft_weight_mean"] = (
                negative_move_mask.sum() / move_denom
            )

        attack_logp_delta = (attack_log_probs - local_log_probs).detach()
        move_logp_delta = (move_log_probs - local_log_probs).detach()
        loss_dict["counterfactual_attack_logp_delta_mean"] = (
            (attack_logp_delta * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_attack_logp_delta_abs_mean"] = (
            (attack_logp_delta.abs() * attack_mask).sum() / attack_denom
        )
        loss_dict["counterfactual_move_logp_delta_mean"] = (
            (move_logp_delta * move_mask).sum() / move_denom
        )
        loss_dict["counterfactual_move_logp_delta_abs_mean"] = (
            (move_logp_delta.abs() * move_mask).sum() / move_denom
        )
        if fused_log_probs is not None:
            fused_logp_delta = (fused_log_probs - local_log_probs).detach()
            loss_dict["counterfactual_fused_logp_delta_mean"] = (
                (fused_logp_delta * policy_mask).sum() / full_denom
            )
            loss_dict["counterfactual_fused_logp_delta_abs_mean"] = (
                (fused_logp_delta.abs() * policy_mask).sum() / full_denom
            )

        action_flip_keys = (
            ("seq_counterfactual_action_flip_fused", "counterfactual_action_flip_fused_rate"),
            ("seq_counterfactual_action_flip_attack_only", "counterfactual_action_flip_attack_only_rate"),
            ("seq_counterfactual_action_flip_move_only", "counterfactual_action_flip_move_only_rate"),
        )
        for seq_key, stat_key in action_flip_keys:
            if seq_key in extra:
                loss_dict[stat_key] = (
                    (extra[seq_key].detach() * policy_mask).sum() / full_denom
                )

        attack_can_mask = extra.get("seq_counterfactual_attack_can_mask", None)
        if attack_can_mask is not None:
            attack_can_mask = attack_can_mask.detach().float() * policy_mask
            attack_can_denom = attack_can_mask.sum().clamp(min=1.0)
            loss_dict["counterfactual_attack_can_ratio"] = (
                attack_can_mask.sum() / full_denom
            )
            target_flip_keys = (
                (
                    "seq_counterfactual_attack_target_flip",
                    "counterfactual_attack_target_flip_rate",
                ),
                (
                    "seq_counterfactual_attack_target_flip_attack_only",
                    "counterfactual_attack_target_flip_attack_only_rate",
                ),
            )
            for seq_key, stat_key in target_flip_keys:
                if seq_key in extra:
                    loss_dict[stat_key] = (
                        (extra[seq_key].detach() * attack_can_mask).sum()
                        / attack_can_denom
                    )

        attack_pair_valid = extra.get("seq_counterfactual_attack_pair_valid", None)
        if attack_pair_valid is not None:
            attack_pair_mask = attack_pair_valid.detach().float() * step_mask
            attack_pair_denom = attack_pair_mask.sum().clamp(min=1.0)
            loss_dict["counterfactual_attack_pair_valid_ratio"] = (
                attack_pair_mask.sum() / step_denom
            )
            agreement_keys = (
                (
                    "seq_counterfactual_attack_target_agreement_local",
                    "counterfactual_attack_target_agreement_local",
                ),
                (
                    "seq_counterfactual_attack_target_agreement_fused",
                    "counterfactual_attack_target_agreement_fused",
                ),
                (
                    "seq_counterfactual_attack_target_agreement_attack_only",
                    "counterfactual_attack_target_agreement_attack_only",
                ),
            )
            agreement_values = {}
            for seq_key, stat_key in agreement_keys:
                if seq_key in extra:
                    agreement_values[stat_key] = (
                        (extra[seq_key].detach() * attack_pair_mask).sum()
                        / attack_pair_denom
                    )
                    loss_dict[stat_key] = agreement_values[stat_key]
            if (
                "counterfactual_attack_target_agreement_local" in agreement_values
                and "counterfactual_attack_target_agreement_fused" in agreement_values
            ):
                loss_dict["counterfactual_attack_target_agreement_gain_fused"] = (
                    agreement_values["counterfactual_attack_target_agreement_fused"]
                    - agreement_values["counterfactual_attack_target_agreement_local"]
                )
            if (
                "counterfactual_attack_target_agreement_local" in agreement_values
                and "counterfactual_attack_target_agreement_attack_only"
                in agreement_values
            ):
                loss_dict["counterfactual_attack_target_agreement_gain_attack_only"] = (
                    agreement_values["counterfactual_attack_target_agreement_attack_only"]
                    - agreement_values["counterfactual_attack_target_agreement_local"]
                )

        return total, loss_dict

    def _process_attack_adv_leverage_loss(
        self,
        total,
        loss_dict,
        extra,
        batch,
        actions=None,
        advantages=None,
        policy_mask=None,
    ):
        if self.attack_adv_leverage_loss_weight <= 0:
            return total, loss_dict
        if actions is None or advantages is None or policy_mask is None:
            return total, loss_dict
        if (
            "seq_counterfactual_local_logp" not in extra
            or "seq_counterfactual_attack_fused_logp" not in extra
        ):
            return total, loss_dict

        action_ids = actions.squeeze(-1)
        semantic_action_offset = int(
            getattr(self.args, "semantic_action_offset", 6)
        )
        attack_mask = (
            (action_ids >= semantic_action_offset).float() * policy_mask
        )
        if "seq_counterfactual_attack_can_mask" in extra:
            attack_mask = (
                attack_mask
                * extra["seq_counterfactual_attack_can_mask"].detach().float()
            )

        real_comm_weight = attack_mask.new_ones(attack_mask.shape)
        if (
            self.attack_adv_leverage_use_real_comm_weight
            and "seq_counterfactual_attack_real_comm_mass" in extra
        ):
            real_comm_weight = extra[
                "seq_counterfactual_attack_real_comm_mass"
            ].detach().float().clamp(min=0.0, max=1.0)

        positive_adv = F.relu(advantages.detach())
        if self.attack_adv_leverage_adv_clip is not None:
            clip_value = float(self.attack_adv_leverage_adv_clip)
            if clip_value > 0:
                positive_adv = positive_adv.clamp(max=clip_value)

        leverage_weight = attack_mask * real_comm_weight * positive_adv
        base_denom = attack_mask.sum().clamp(min=1.0)
        if self.attack_adv_leverage_fixed_denom:
            loss_denom = base_denom
        else:
            loss_denom = leverage_weight.sum().clamp(min=1.0)

        local_log_probs = extra["seq_counterfactual_local_logp"].detach()
        attack_fused_log_probs = extra["seq_counterfactual_attack_fused_logp"]
        logp_delta = attack_fused_log_probs - local_log_probs
        margin = logp_delta.new_tensor(float(self.attack_adv_leverage_margin))
        hinge = F.relu(margin - logp_delta)
        if self.attack_adv_leverage_loss_mode == "huber":
            beta = hinge.new_tensor(float(self.attack_adv_leverage_huber_beta))
            leverage_penalty = th.where(
                hinge < beta,
                0.5 * hinge.pow(2) / beta,
                hinge - 0.5 * beta,
            )
        elif self.attack_adv_leverage_loss_mode == "linear":
            leverage_penalty = hinge
        else:
            leverage_penalty = hinge.pow(2)
        leverage_loss = self.attack_adv_leverage_loss_weight * (
            (leverage_penalty * leverage_weight).sum() / loss_denom
        )
        total = total + leverage_loss

        logp_delta_detached = logp_delta.detach()
        hinge_detached = hinge.detach()
        penalty_detached = leverage_penalty.detach()
        weighted_denom = leverage_weight.detach().sum().clamp(min=1.0)
        loss_dict["attack_adv_leverage_loss"] = leverage_loss.detach()
        loss_dict["attack_adv_leverage_logp_delta_mean"] = (
            (logp_delta_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_logp_delta_abs_mean"] = (
            (logp_delta_detached.abs() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_hinge_mean"] = (
            (hinge_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_penalty_mean"] = (
            (penalty_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_weight_mean"] = (
            leverage_weight.detach().sum() / base_denom
        )
        loss_dict["attack_adv_leverage_positive_ratio"] = (
            ((positive_adv > 0).float() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_real_comm_mass"] = (
            (real_comm_weight * attack_mask).sum() / base_denom
        )
        loss_dict["attack_adv_leverage_margin"] = margin.detach()
        loss_dict["attack_adv_leverage_huber_beta"] = logp_delta.new_tensor(
            float(self.attack_adv_leverage_huber_beta)
        ).detach()
        loss_dict["attack_adv_leverage_weighted_logp_delta_mean"] = (
            (logp_delta_detached * leverage_weight.detach()).sum()
            / weighted_denom
        )
        loss_dict["attack_adv_leverage_weighted_penalty_mean"] = (
            (penalty_detached * leverage_weight.detach()).sum()
            / weighted_denom
        )
        return total, loss_dict

    def _process_attack_causal_leverage_loss(
        self,
        total,
        loss_dict,
        extra,
        batch,
        actions=None,
        advantages=None,
        policy_mask=None,
    ):
        if (
            self.attack_causal_leverage_loss_weight <= 0
            and self.attack_causal_stability_loss_weight <= 0
            and self.attack_causal_bad_loss_weight <= 0
        ):
            return total, loss_dict
        if actions is None or advantages is None or policy_mask is None:
            return total, loss_dict

        required_keys = (
            "seq_counterfactual_local_attack_target_logp",
            "seq_counterfactual_attack_fused_target_logp",
            "seq_counterfactual_attack_local_target_prob",
            "seq_counterfactual_attack_peer_target_prob",
            "seq_counterfactual_attack_local_target_match",
            "seq_counterfactual_attack_target_available",
        )
        if any(key not in extra for key in required_keys):
            return total, loss_dict

        action_ids = actions.squeeze(-1)
        semantic_action_offset = int(
            getattr(self.args, "semantic_action_offset", 6)
        )
        attack_mask = (
            (action_ids >= semantic_action_offset).float() * policy_mask
        )
        if "seq_counterfactual_attack_can_mask" in extra:
            attack_mask = (
                attack_mask
                * extra["seq_counterfactual_attack_can_mask"].detach().float()
            )
        target_available = extra[
            "seq_counterfactual_attack_target_available"
        ].detach().float()
        attack_mask = attack_mask * target_available
        base_denom = attack_mask.sum().clamp(min=1.0)

        real_comm_weight = attack_mask.new_ones(attack_mask.shape)
        if (
            self.attack_causal_leverage_use_real_comm_weight
            and "seq_counterfactual_attack_real_comm_mass" in extra
        ):
            real_comm_weight = extra[
                "seq_counterfactual_attack_real_comm_mass"
            ].detach().float().clamp(min=0.0, max=1.0)

        positive_adv = F.relu(advantages.detach())
        if self.attack_causal_leverage_adv_clip is not None:
            clip_value = float(self.attack_causal_leverage_adv_clip)
            if clip_value > 0:
                positive_adv = positive_adv.clamp(max=clip_value)

        local_target_prob = extra[
            "seq_counterfactual_attack_local_target_prob"
        ].detach().float().clamp(min=0.0, max=1.0)
        peer_target_prob = extra[
            "seq_counterfactual_attack_peer_target_prob"
        ].detach().float().clamp(min=0.0, max=1.0)
        local_target_match = extra[
            "seq_counterfactual_attack_local_target_match"
        ].detach().float().clamp(min=0.0, max=1.0)

        mismatch_score = (1.0 - local_target_match).clamp(min=0.0, max=1.0)
        if self.attack_causal_local_target_prob_threshold > 0:
            local_threshold = local_target_prob.new_tensor(
                float(self.attack_causal_local_target_prob_threshold)
            )
            low_local_support = (
                (local_threshold - local_target_prob)
                / local_threshold.clamp(min=1e-6)
            ).clamp(min=0.0, max=1.0)
        else:
            low_local_support = th.zeros_like(local_target_prob)
        local_need_score = th.maximum(mismatch_score, low_local_support)

        if self.attack_causal_peer_support_threshold > 0:
            peer_threshold = peer_target_prob.new_tensor(
                float(self.attack_causal_peer_support_threshold)
            )
            peer_support_score = (
                peer_target_prob / peer_threshold.clamp(min=1e-6)
            ).clamp(min=0.0, max=1.0)
        else:
            peer_support_score = th.ones_like(peer_target_prob)

        causal_weight = (
            attack_mask
            * real_comm_weight
            * positive_adv
            * local_need_score
            * peer_support_score
        )
        if self.attack_causal_leverage_fixed_denom:
            loss_denom = base_denom
        else:
            loss_denom = causal_weight.sum().clamp(min=1.0)

        local_target_logp = extra[
            "seq_counterfactual_local_attack_target_logp"
        ].detach()
        fused_target_logp = extra["seq_counterfactual_attack_fused_target_logp"]
        target_logp_delta = fused_target_logp - local_target_logp
        margin = target_logp_delta.new_tensor(
            float(self.attack_causal_leverage_margin)
        )
        hinge = F.relu(margin - target_logp_delta)
        beta = hinge.new_tensor(float(self.attack_causal_leverage_huber_beta))
        leverage_penalty = th.where(
            hinge < beta,
            0.5 * hinge.pow(2) / beta,
            hinge - 0.5 * beta,
        )

        leverage_loss = self.attack_causal_leverage_loss_weight * (
            (leverage_penalty * causal_weight).sum() / loss_denom
        )
        total = total + leverage_loss

        bad_loss = attack_mask.new_zeros(())
        bad_weight = attack_mask.new_zeros(attack_mask.shape)
        bad_chosen_delta = target_logp_delta
        bad_peer_delta = attack_mask.new_zeros(attack_mask.shape)
        bad_chosen_hinge = attack_mask.new_zeros(attack_mask.shape)
        bad_peer_hinge = attack_mask.new_zeros(attack_mask.shape)
        bad_peer_support_score = attack_mask.new_zeros(attack_mask.shape)
        bad_peer_top1_prob = attack_mask.new_zeros(attack_mask.shape)
        bad_peer_conflict = attack_mask.new_zeros(attack_mask.shape)
        bad_peer_available = attack_mask.new_zeros(attack_mask.shape)
        bad_stability_block = attack_mask.new_zeros(attack_mask.shape)
        negative_adv = F.relu(-advantages.detach())
        if self.attack_causal_bad_adv_clip is not None:
            bad_clip_value = float(self.attack_causal_bad_adv_clip)
            if bad_clip_value > 0:
                negative_adv = negative_adv.clamp(max=bad_clip_value)
        if (
            self.attack_causal_bad_loss_weight > 0
            and "seq_counterfactual_local_attack_peer_top1_logp" in extra
            and "seq_counterfactual_attack_fused_peer_top1_logp" in extra
            and "seq_counterfactual_attack_peer_top1_prob" in extra
            and "seq_counterfactual_attack_peer_top1_available" in extra
            and "seq_counterfactual_attack_peer_top1_match_chosen" in extra
        ):
            bad_peer_top1_prob = extra[
                "seq_counterfactual_attack_peer_top1_prob"
            ].detach().float().clamp(min=0.0, max=1.0)
            bad_peer_available = extra[
                "seq_counterfactual_attack_peer_top1_available"
            ].detach().float().clamp(min=0.0, max=1.0)
            bad_peer_conflict = (
                1.0
                - extra[
                    "seq_counterfactual_attack_peer_top1_match_chosen"
                ].detach().float().clamp(min=0.0, max=1.0)
            )
            if self.attack_causal_bad_peer_support_threshold > 0:
                bad_peer_threshold = bad_peer_top1_prob.new_tensor(
                    float(self.attack_causal_bad_peer_support_threshold)
                )
                bad_peer_support_score = (
                    (bad_peer_top1_prob - bad_peer_threshold)
                    / (1.0 - bad_peer_threshold).clamp(min=1e-6)
                ).clamp(min=0.0, max=1.0)
            else:
                bad_peer_support_score = th.ones_like(bad_peer_top1_prob)

            # The bad-action branch focuses on cases where the sampled negative-advantage
            # attack is also the local policy's own top target. That is the closest
            # on-policy proxy for "local policy would make this bad choice."
            bad_weight = (
                attack_mask
                * real_comm_weight
                * negative_adv
                * local_target_match
                * local_target_prob
                * bad_peer_conflict
                * bad_peer_available
                * bad_peer_support_score
            )
            bad_stability_block = (bad_weight.detach() > 0).float()
            bad_margin = target_logp_delta.new_tensor(
                float(self.attack_causal_bad_margin)
            )
            bad_chosen_hinge = F.relu(bad_margin + bad_chosen_delta)
            bad_beta = bad_chosen_hinge.new_tensor(
                float(self.attack_causal_bad_huber_beta)
            )
            bad_chosen_penalty = th.where(
                bad_chosen_hinge < bad_beta,
                0.5 * bad_chosen_hinge.pow(2) / bad_beta,
                bad_chosen_hinge - 0.5 * bad_beta,
            )

            bad_peer_local_logp = extra[
                "seq_counterfactual_local_attack_peer_top1_logp"
            ].detach()
            bad_peer_fused_logp = extra[
                "seq_counterfactual_attack_fused_peer_top1_logp"
            ]
            bad_peer_delta = bad_peer_fused_logp - bad_peer_local_logp
            bad_peer_hinge = F.relu(bad_margin - bad_peer_delta)
            bad_peer_penalty = th.where(
                bad_peer_hinge < bad_beta,
                0.5 * bad_peer_hinge.pow(2) / bad_beta,
                bad_peer_hinge - 0.5 * bad_beta,
            )
            bad_combined_penalty = (
                bad_chosen_penalty
                + float(self.attack_causal_bad_peer_boost_weight)
                * bad_peer_penalty
            )
            if self.attack_causal_leverage_fixed_denom:
                bad_denom = base_denom
            else:
                bad_denom = bad_weight.sum().clamp(min=1.0)
            bad_loss = self.attack_causal_bad_loss_weight * (
                (bad_combined_penalty * bad_weight).sum() / bad_denom
            )
            total = total + bad_loss

        stability_kl = extra.get("seq_counterfactual_attack_stability_kl", None)
        if stability_kl is not None and self.attack_causal_stability_loss_weight > 0:
            stability_weight = (
                attack_mask
                * real_comm_weight
                * (1.0 - local_need_score).clamp(min=0.0, max=1.0)
                * (1.0 - bad_stability_block).clamp(min=0.0, max=1.0)
            )
            stability_loss = self.attack_causal_stability_loss_weight * (
                (stability_kl * stability_weight).sum() / base_denom
            )
            total = total + stability_loss
        else:
            stability_weight = attack_mask.new_zeros(attack_mask.shape)
            stability_loss = attack_mask.new_zeros(())

        detached_weight = causal_weight.detach()
        weighted_denom = detached_weight.sum().clamp(min=1.0)
        detached_stability_weight = stability_weight.detach()
        stability_denom = detached_stability_weight.sum().clamp(min=1.0)
        target_logp_delta_detached = target_logp_delta.detach()
        hinge_detached = hinge.detach()
        penalty_detached = leverage_penalty.detach()
        detached_bad_weight = bad_weight.detach()
        bad_weighted_denom = detached_bad_weight.sum().clamp(min=1.0)

        loss_dict["attack_causal_leverage_loss"] = leverage_loss.detach()
        loss_dict["attack_causal_bad_loss"] = bad_loss.detach()
        loss_dict["attack_causal_stability_loss"] = stability_loss.detach()
        loss_dict["attack_causal_target_logp_delta_mean"] = (
            (target_logp_delta_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_target_logp_delta_abs_mean"] = (
            (target_logp_delta_detached.abs() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_weighted_target_logp_delta_mean"] = (
            (target_logp_delta_detached * detached_weight).sum() / weighted_denom
        )
        loss_dict["attack_causal_hinge_mean"] = (
            (hinge_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_penalty_mean"] = (
            (penalty_detached * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_weight_mean"] = detached_weight.sum() / base_denom
        loss_dict["attack_causal_positive_ratio"] = (
            ((positive_adv > 0).float() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_real_comm_mass"] = (
            (real_comm_weight * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_local_need_score"] = (
            (local_need_score * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_mismatch_score"] = (
            (mismatch_score * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_low_local_support"] = (
            (low_local_support * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_peer_support_score"] = (
            (peer_support_score * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_local_target_prob"] = (
            (local_target_prob * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_peer_target_prob"] = (
            (peer_target_prob * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_local_target_match"] = (
            (local_target_match * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_target_available"] = (
            (target_available * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)
        )
        loss_dict["attack_causal_margin"] = margin.detach()
        loss_dict["attack_causal_huber_beta"] = target_logp_delta.new_tensor(
            float(self.attack_causal_leverage_huber_beta)
        ).detach()
        loss_dict["attack_causal_local_target_prob_threshold"] = (
            target_logp_delta.new_tensor(
                float(self.attack_causal_local_target_prob_threshold)
            ).detach()
        )
        loss_dict["attack_causal_peer_support_threshold"] = (
            target_logp_delta.new_tensor(
                float(self.attack_causal_peer_support_threshold)
            ).detach()
        )
        loss_dict["attack_causal_safe_weight_mean"] = (
            detached_stability_weight.sum() / base_denom
        )
        loss_dict["attack_causal_bad_weight_mean"] = (
            detached_bad_weight.sum() / base_denom
        )
        loss_dict["attack_causal_bad_negative_ratio"] = (
            ((negative_adv > 0).float() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_peer_top1_prob"] = (
            (bad_peer_top1_prob * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_peer_support_score"] = (
            (bad_peer_support_score * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_peer_conflict_ratio"] = (
            (bad_peer_conflict * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_peer_available"] = (
            (bad_peer_available * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_chosen_logp_delta_mean"] = (
            (bad_chosen_delta.detach() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_weighted_chosen_logp_delta_mean"] = (
            (bad_chosen_delta.detach() * detached_bad_weight).sum()
            / bad_weighted_denom
        )
        loss_dict["attack_causal_bad_peer_logp_delta_mean"] = (
            (bad_peer_delta.detach() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_weighted_peer_logp_delta_mean"] = (
            (bad_peer_delta.detach() * detached_bad_weight).sum()
            / bad_weighted_denom
        )
        loss_dict["attack_causal_bad_chosen_hinge_mean"] = (
            (bad_chosen_hinge.detach() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_peer_hinge_mean"] = (
            (bad_peer_hinge.detach() * attack_mask).sum() / base_denom
        )
        loss_dict["attack_causal_bad_margin"] = target_logp_delta.new_tensor(
            float(self.attack_causal_bad_margin)
        ).detach()
        loss_dict["attack_causal_bad_huber_beta"] = target_logp_delta.new_tensor(
            float(self.attack_causal_bad_huber_beta)
        ).detach()
        loss_dict["attack_causal_bad_peer_support_threshold"] = (
            target_logp_delta.new_tensor(
                float(self.attack_causal_bad_peer_support_threshold)
            ).detach()
        )
        loss_dict["attack_causal_bad_peer_boost_weight"] = (
            target_logp_delta.new_tensor(
                float(self.attack_causal_bad_peer_boost_weight)
            ).detach()
        )
        if stability_kl is not None:
            stability_kl_detached = stability_kl.detach()
            loss_dict["attack_causal_stability_kl_mean"] = (
                (stability_kl_detached * attack_mask).sum() / base_denom
            )
            loss_dict["attack_causal_weighted_stability_kl_mean"] = (
                (stability_kl_detached * detached_stability_weight).sum()
                / stability_denom
            )
        return total, loss_dict

    def _process_attack_peer_conflict_margin_leverage_loss(
        self,
        total,
        loss_dict,
        extra,
        batch,
        actions=None,
        policy_mask=None,
    ):
        if (
            self.attack_peer_conflict_margin_leverage_loss_weight <= 0
            and self.attack_peer_conflict_attack_only_margin_loss_weight <= 0
        ):
            return total, loss_dict
        if policy_mask is None:
            return total, loss_dict

        base_required_keys = (
            "seq_counterfactual_attack_peer_top1_prob",
            "seq_counterfactual_attack_local_top1_prob",
            "seq_counterfactual_attack_peer_valid_mask",
            "seq_counterfactual_attack_can_mask",
            "seq_counterfactual_attack_peer_top1",
            "seq_counterfactual_attack_local_top1",
        )
        if any(key not in extra for key in base_required_keys):
            return total, loss_dict

        fused_required_keys = (
            "seq_counterfactual_attack_fused_peer_top1_logp",
            "seq_counterfactual_attack_fused_local_top1_logp",
        )
        attack_only_required_keys = (
            "seq_counterfactual_attack_attack_only_peer_top1_logp",
            "seq_counterfactual_attack_attack_only_local_top1_logp",
        )
        fused_margin_enabled = (
            self.attack_peer_conflict_margin_leverage_loss_weight > 0
            and all(key in extra for key in fused_required_keys)
        )
        attack_only_margin_enabled = (
            self.attack_peer_conflict_attack_only_margin_loss_weight > 0
            and all(key in extra for key in attack_only_required_keys)
        )
        if not fused_margin_enabled and not attack_only_margin_enabled:
            return total, loss_dict

        attack_can = extra["seq_counterfactual_attack_can_mask"].detach().float()
        peer_valid = extra[
            "seq_counterfactual_attack_peer_valid_mask"
        ].detach().float()
        # Do not condition this auxiliary on the sampled/chosen action. The
        # diagnostic showed chosen attacks are almost always local-top1, which
        # would recreate the on-policy lock we are trying to break.
        base_mask = policy_mask * attack_can * peer_valid
        base_denom = base_mask.sum().clamp(min=1.0)

        peer_top1 = extra["seq_counterfactual_attack_peer_top1"].detach().long()
        local_top1 = extra["seq_counterfactual_attack_local_top1"].detach().long()
        conflict = (peer_top1 != local_top1).float() * base_mask

        peer_conf = extra[
            "seq_counterfactual_attack_peer_top1_prob"
        ].detach().float().clamp(min=0.0, max=1.0)
        local_conf = extra[
            "seq_counterfactual_attack_local_top1_prob"
        ].detach().float().clamp(min=0.0, max=1.0)
        threshold = float(self.attack_peer_conflict_peer_support_threshold)
        if threshold > 0:
            threshold_tensor = peer_conf.new_tensor(threshold)
            peer_support_score = (
                (peer_conf - threshold_tensor)
                / (1.0 - threshold_tensor).clamp(min=1e-6)
            ).clamp(min=0.0, max=1.0)
        else:
            peer_support_score = th.ones_like(peer_conf)

        real_comm_weight = base_mask.new_ones(base_mask.shape)
        if (
            self.attack_peer_conflict_use_real_comm_weight
            and "seq_counterfactual_attack_real_comm_mass" in extra
        ):
            real_comm_weight = extra[
                "seq_counterfactual_attack_real_comm_mass"
            ].detach().float().clamp(min=0.0, max=1.0)

        leverage_weight = (
            conflict
            * real_comm_weight
            * peer_support_score
            * local_conf
        )
        fused_local_uncertainty_weight = th.ones_like(leverage_weight)
        if self.attack_peer_conflict_fused_local_conf_max is not None:
            local_conf_max = local_conf.new_tensor(
                float(self.attack_peer_conflict_fused_local_conf_max)
            )
            min_weight = local_conf.new_tensor(
                float(self.attack_peer_conflict_fused_local_uncertainty_min_weight)
            )
            local_uncertainty = (
                (local_conf_max - local_conf)
                / local_conf_max.clamp(min=1e-6)
            ).clamp(min=0.0, max=1.0)
            fused_local_uncertainty_weight = (
                min_weight + (1.0 - min_weight) * local_uncertainty
            )
        fused_leverage_weight = leverage_weight * fused_local_uncertainty_weight
        if self.attack_peer_conflict_fixed_denom:
            fused_loss_denom = base_denom
            attack_only_loss_denom = base_denom
        else:
            fused_loss_denom = fused_leverage_weight.sum().clamp(min=1.0)
            attack_only_loss_denom = leverage_weight.sum().clamp(min=1.0)

        detached_weight = leverage_weight.detach()
        weighted_denom = detached_weight.sum().clamp(min=1.0)
        fused_detached_weight = fused_leverage_weight.detach()
        fused_weighted_denom = fused_detached_weight.sum().clamp(min=1.0)
        conflict_denom = conflict.sum().clamp(min=1.0)
        target_margin = peer_conf.new_tensor(float(self.attack_peer_conflict_margin))
        beta = peer_conf.new_tensor(float(self.attack_peer_conflict_margin_huber_beta))

        def _margin_terms(peer_logp, local_logp):
            margin = peer_logp - local_logp
            hinge = F.relu(target_margin - margin)
            penalty = th.where(
                hinge < beta,
                0.5 * hinge.pow(2) / beta,
                hinge - 0.5 * beta,
            )
            return margin, hinge, penalty

        total_margin_loss = target_margin.new_zeros(())

        if fused_margin_enabled:
            fused_peer_logp = extra[
                "seq_counterfactual_attack_fused_peer_top1_logp"
            ]
            fused_local_logp = extra[
                "seq_counterfactual_attack_fused_local_top1_logp"
            ]
            fused_margin, hinge, margin_penalty = _margin_terms(
                fused_peer_logp, fused_local_logp
            )
            margin_loss = self.attack_peer_conflict_margin_leverage_loss_weight * (
                (margin_penalty * fused_leverage_weight).sum() / fused_loss_denom
            )
            total_margin_loss = total_margin_loss + margin_loss

            fused_margin_detached = fused_margin.detach()
            hinge_detached = hinge.detach()
            margin_penalty_detached = margin_penalty.detach()

            loss_dict["attack_peer_conflict_margin_loss"] = margin_loss.detach()
            loss_dict["attack_peer_conflict_margin_logp_delta_mean"] = (
                (fused_margin_detached * conflict).sum() / conflict_denom
            )
            loss_dict["attack_peer_conflict_margin_weighted_logp_delta_mean"] = (
                (fused_margin_detached * fused_detached_weight).sum()
                / fused_weighted_denom
            )
            loss_dict["attack_peer_conflict_margin_hinge_mean"] = (
                (hinge_detached * conflict).sum() / conflict_denom
            )
            loss_dict["attack_peer_conflict_margin_weighted_hinge_mean"] = (
                (hinge_detached * fused_detached_weight).sum()
                / fused_weighted_denom
            )
            loss_dict["attack_peer_conflict_margin_penalty_mean"] = (
                (margin_penalty_detached * conflict).sum() / conflict_denom
            )

        if attack_only_margin_enabled:
            attack_only_peer_logp = extra[
                "seq_counterfactual_attack_attack_only_peer_top1_logp"
            ]
            attack_only_local_logp = extra[
                "seq_counterfactual_attack_attack_only_local_top1_logp"
            ]
            (
                attack_only_margin,
                attack_only_hinge,
                attack_only_margin_penalty,
            ) = _margin_terms(attack_only_peer_logp, attack_only_local_logp)
            attack_only_margin_loss = (
                self.attack_peer_conflict_attack_only_margin_loss_weight
                * (
                    (attack_only_margin_penalty * leverage_weight).sum()
                    / attack_only_loss_denom
                )
            )
            total_margin_loss = total_margin_loss + attack_only_margin_loss

            attack_only_margin_detached = attack_only_margin.detach()
            attack_only_hinge_detached = attack_only_hinge.detach()
            attack_only_margin_penalty_detached = attack_only_margin_penalty.detach()

            loss_dict[
                "attack_peer_conflict_attack_only_margin_loss"
            ] = attack_only_margin_loss.detach()
            loss_dict[
                "attack_peer_conflict_attack_only_logp_delta_mean"
            ] = (
                (attack_only_margin_detached * conflict).sum() / conflict_denom
            )
            loss_dict[
                "attack_peer_conflict_attack_only_weighted_logp_delta_mean"
            ] = (
                (attack_only_margin_detached * detached_weight).sum()
                / weighted_denom
            )
            loss_dict["attack_peer_conflict_attack_only_hinge_mean"] = (
                (attack_only_hinge_detached * conflict).sum() / conflict_denom
            )
            loss_dict[
                "attack_peer_conflict_attack_only_weighted_hinge_mean"
            ] = (
                (attack_only_hinge_detached * detached_weight).sum()
                / weighted_denom
            )
            loss_dict[
                "attack_peer_conflict_attack_only_penalty_mean"
            ] = (
                (attack_only_margin_penalty_detached * conflict).sum()
                / conflict_denom
            )

        total = total + total_margin_loss

        loss_dict[
            "attack_peer_conflict_margin_total_loss"
        ] = total_margin_loss.detach()
        loss_dict["attack_peer_conflict_margin_weight_mean"] = (
            detached_weight.sum() / base_denom
        )
        loss_dict["attack_peer_conflict_fused_margin_weight_mean"] = (
            fused_detached_weight.sum() / base_denom
        )
        loss_dict[
            "attack_peer_conflict_fused_local_uncertainty_weight_mean"
        ] = (
            (fused_local_uncertainty_weight.detach() * conflict).sum()
            / conflict_denom
        )
        loss_dict[
            "attack_peer_conflict_fused_local_uncertainty_weighted_local_conf"
        ] = (
            (local_conf * fused_detached_weight).sum() / fused_weighted_denom
        )
        loss_dict["attack_peer_conflict_margin_conflict_rate"] = (
            conflict.sum() / base_denom
        )
        loss_dict["attack_peer_conflict_margin_peer_conf"] = (
            (peer_conf * base_mask).sum() / base_denom
        )
        loss_dict["attack_peer_conflict_margin_local_conf"] = (
            (local_conf * base_mask).sum() / base_denom
        )
        loss_dict["attack_peer_conflict_margin_peer_support_score"] = (
            (peer_support_score * base_mask).sum() / base_denom
        )
        loss_dict["attack_peer_conflict_margin_real_comm_mass"] = (
            (real_comm_weight * base_mask).sum() / base_denom
        )
        loss_dict["attack_peer_conflict_margin"] = target_margin.detach()
        loss_dict["attack_peer_conflict_margin_huber_beta"] = beta.detach()
        loss_dict["attack_peer_conflict_peer_support_threshold"] = (
            target_margin.new_tensor(threshold).detach()
        )
        fused_local_conf_max = (
            -1.0
            if self.attack_peer_conflict_fused_local_conf_max is None
            else float(self.attack_peer_conflict_fused_local_conf_max)
        )
        loss_dict["attack_peer_conflict_fused_local_conf_max"] = (
            target_margin.new_tensor(fused_local_conf_max).detach()
        )
        loss_dict[
            "attack_peer_conflict_fused_local_uncertainty_min_weight"
        ] = (
            target_margin.new_tensor(
                float(self.attack_peer_conflict_fused_local_uncertainty_min_weight)
            ).detach()
        )

        if "seq_counterfactual_attack_fused_top1" in extra:
            fused_top1 = extra["seq_counterfactual_attack_fused_top1"].detach().long()
            loss_dict["attack_peer_conflict_fused_follow_peer_rate"] = (
                ((fused_top1 == peer_top1).float() * conflict).sum()
                / conflict_denom
            )
            loss_dict["attack_peer_conflict_fused_stay_local_rate"] = (
                ((fused_top1 == local_top1).float() * conflict).sum()
                / conflict_denom
            )
        if "seq_counterfactual_attack_attack_only_top1" in extra:
            attack_only_top1 = extra[
                "seq_counterfactual_attack_attack_only_top1"
            ].detach().long()
            loss_dict["attack_peer_conflict_attack_only_follow_peer_rate"] = (
                ((attack_only_top1 == peer_top1).float() * conflict).sum()
                / conflict_denom
            )
            loss_dict["attack_peer_conflict_attack_only_stay_local_rate"] = (
                ((attack_only_top1 == local_top1).float() * conflict).sum()
                / conflict_denom
            )

        return total, loss_dict

    def _normalize_usegate_gain(self, gain, gain_mask, stream_name):
        denom = gain_mask.sum().clamp(min=1.0)
        current_abs_mean = ((gain.detach().abs() * gain_mask).sum() / denom).item()
        decay = self.counterfactual_usegate_gain_norm_ema_decay

        if stream_name == "attack":
            prev_ema = self._attack_gain_abs_ema
            next_ema = (
                current_abs_mean
                if prev_ema is None
                else decay * prev_ema + (1.0 - decay) * current_abs_mean
            )
            self._attack_gain_abs_ema = next_ema
            eps = self.counterfactual_usegate_attack_gain_norm_eps
        elif stream_name == "move":
            prev_ema = self._move_gain_abs_ema
            next_ema = (
                current_abs_mean
                if prev_ema is None
                else decay * prev_ema + (1.0 - decay) * current_abs_mean
            )
            self._move_gain_abs_ema = next_ema
            eps = self.counterfactual_usegate_move_gain_norm_eps
        else:
            raise ValueError("Unsupported stream_name '{}'".format(stream_name))

        scale = max(eps, float(next_ema))
        return gain / scale, gain.new_tensor(scale)

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

    def _compute_comm_warmup_factor(self):
        if self.comm_warmup_steps <= 0:
            return 1.0
        delay_steps = max(0, self.comm_warmup_delay_steps)
        shifted_t = float(self._t_env) - float(delay_steps)
        progress = float(max(0.0, min(1.0, shifted_t / float(self.comm_warmup_steps))))
        nonlinear_progress = progress ** self.comm_warmup_exponent
        return (
            self.comm_warmup_start_factor
            + (self.comm_warmup_end_factor - self.comm_warmup_start_factor) * nonlinear_progress
        )

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

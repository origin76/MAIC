import torch as th
import torch.nn.functional as F

from .budgeted_sparse_mappo_learner import BudgetedSparseMAPPOLearner


class BudgetedSparseMAPPOSemanticHeadV2Learner(BudgetedSparseMAPPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super(BudgetedSparseMAPPOSemanticHeadV2Learner, self).__init__(mac, scheme, logger, args)
        self.semantic_action_offset = getattr(args, "semantic_action_offset", 6)
        self.semantic_action_loss_weight = getattr(args, "semantic_action_loss_weight", 0.005)
        self.semantic_state_loss_weight = getattr(args, "semantic_state_loss_weight", 0.01)
        self.semantic_state_feature_indices = list(getattr(args, "semantic_state_feature_indices", [0, 1, 2, 3, 4]))

        self.semantic_action_aux_warmup_steps = getattr(args, "semantic_action_aux_warmup_steps", 0)
        self.semantic_action_aux_decay_start = getattr(args, "semantic_action_aux_decay_start", None)
        self.semantic_action_aux_decay_end = getattr(args, "semantic_action_aux_decay_end", None)
        self.semantic_action_aux_final_scale = getattr(args, "semantic_action_aux_final_scale", 1.0)

        self.semantic_state_aux_warmup_steps = getattr(args, "semantic_state_aux_warmup_steps", 0)
        self.semantic_state_aux_decay_start = getattr(args, "semantic_state_aux_decay_start", None)
        self.semantic_state_aux_decay_end = getattr(args, "semantic_state_aux_decay_end", None)
        self.semantic_state_aux_final_scale = getattr(args, "semantic_state_aux_final_scale", 1.0)

        self.semantic_action_aux_scale = 1.0
        self.semantic_state_aux_scale = 1.0

        self.semantic_action_dim = args.n_actions - self.semantic_action_offset
        if self.semantic_action_dim <= 0:
            raise ValueError("semantic_action_offset must be smaller than n_actions")
        if len(self.semantic_state_feature_indices) != getattr(args, "semantic_state_dim", len(self.semantic_state_feature_indices)):
            raise ValueError("semantic_state_dim must match len(semantic_state_feature_indices)")

        self._init_obs_layout(scheme, args)

    def train(self, batch, t_env, episode_num):
        self.semantic_action_aux_scale = self._get_aux_scale(
            t_env,
            self.semantic_action_aux_warmup_steps,
            self.semantic_action_aux_decay_start,
            self.semantic_action_aux_decay_end,
            self.semantic_action_aux_final_scale,
        )
        self.semantic_state_aux_scale = self._get_aux_scale(
            t_env,
            self.semantic_state_aux_warmup_steps,
            self.semantic_state_aux_decay_start,
            self.semantic_state_aux_decay_end,
            self.semantic_state_aux_final_scale,
        )
        super().train(batch, t_env, episode_num)

    def _forward_policy(self, batch, prepare_for_logging=False):
        outputs = []
        loss_items = []
        logs = []
        semantic_action_preds = []
        semantic_action_attns = []
        semantic_state_preds = []
        semantic_state_attns = []

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
            if "semantic_action_pred" in extra:
                semantic_action_preds.append(extra["semantic_action_pred"])
                del extra["semantic_action_pred"]
            if "semantic_action_attn" in extra:
                semantic_action_attns.append(extra["semantic_action_attn"])
                del extra["semantic_action_attn"]
            if "semantic_state_pred" in extra:
                semantic_state_preds.append(extra["semantic_state_pred"])
                del extra["semantic_state_pred"]
            if "semantic_state_attn" in extra:
                semantic_state_attns.append(extra["semantic_state_attn"])
                del extra["semantic_state_attn"]

            loss_items.append(extra)

        policy = th.stack(outputs, dim=1)
        merged = self._merge_extra_items(loss_items, logs)
        if len(semantic_action_preds) > 0:
            merged["semantic_action_pred_seq"] = th.stack(semantic_action_preds, dim=1)
        if len(semantic_action_attns) > 0:
            merged["semantic_action_attn_seq"] = th.stack(semantic_action_attns, dim=1)
        if len(semantic_state_preds) > 0:
            merged["semantic_state_pred_seq"] = th.stack(semantic_state_preds, dim=1)
        if len(semantic_state_attns) > 0:
            merged["semantic_state_attn_seq"] = th.stack(semantic_state_attns, dim=1)
        return policy, merged

    def _process_extra_losses(self, extra, batch):
        total, loss_dict = super()._process_extra_losses(extra, batch)

        transition_mask = batch["filled"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()
        transition_mask[:, 1:] = transition_mask[:, 1:] * (1 - terminated[:, :-1])
        agent_mask = transition_mask.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.n_agents)

        semantic_action_pred = extra.get("semantic_action_pred_seq", None)
        semantic_action_attn = extra.get("semantic_action_attn_seq", None)
        if semantic_action_pred is not None and semantic_action_attn is not None and self.semantic_action_loss_weight > 0:
            action_targets = self._build_attack_action_targets(batch["actions"][:, :-1]).detach()
            weighted_action_targets = th.matmul(semantic_action_attn.detach(), action_targets)
            action_loss, action_logs = self._compute_action_aux_loss(
                semantic_action_pred,
                weighted_action_targets,
                transition_mask,
                agent_mask,
            )
            total = total + action_loss
            loss_dict.update(action_logs)

        semantic_state_pred = extra.get("semantic_state_pred_seq", None)
        semantic_state_attn = extra.get("semantic_state_attn_seq", None)
        if semantic_state_pred is not None and semantic_state_attn is not None and self.semantic_state_loss_weight > 0:
            ally_state_targets = self._build_ally_state_targets(batch["obs"][:, :-1]).detach()
            aligned_state_attn = self._drop_self_attention(semantic_state_attn.detach())
            weighted_state_targets = th.sum(aligned_state_attn.unsqueeze(-1) * ally_state_targets, dim=3)
            state_loss, state_logs = self._compute_state_aux_loss(
                semantic_state_pred,
                weighted_state_targets,
                transition_mask,
                agent_mask,
            )
            total = total + state_loss
            loss_dict.update(state_logs)

        return total, loss_dict

    def _compute_action_aux_loss(self, pred, target, transition_mask, agent_mask):
        target_mask = transition_mask.unsqueeze(-1).expand_as(target)
        raw_action_loss = F.mse_loss(pred, target, reduction="none")
        raw_action_loss = (raw_action_loss * target_mask).sum() / target_mask.sum().clamp(min=1.0)

        scaled_action_loss = raw_action_loss * self.semantic_action_loss_weight * self.semantic_action_aux_scale
        denominator = agent_mask.sum().clamp(min=1.0)
        target_attack_mass = (target.detach().sum(dim=-1) * agent_mask).sum() / denominator
        pred_attack_mass = (pred.detach().sum(dim=-1) * agent_mask).sum() / denominator
        positive_ratio = (((target.detach().sum(dim=-1) > 0).float()) * agent_mask).sum() / denominator

        logs = {
            "semantic_action_loss": scaled_action_loss.detach(),
            "semantic_action_raw_mse": raw_action_loss.detach(),
            "semantic_action_target_attack_mass": target_attack_mass.detach(),
            "semantic_action_pred_attack_mass": pred_attack_mass.detach(),
            "semantic_action_positive_ratio": positive_ratio.detach(),
            "semantic_action_aux_scale": pred.new_tensor(self.semantic_action_aux_scale),
        }
        return scaled_action_loss, logs

    def _compute_state_aux_loss(self, pred, target, transition_mask, agent_mask):
        target_mask = transition_mask.unsqueeze(-1).expand_as(target)
        raw_state_loss = F.smooth_l1_loss(pred, target, reduction="none")
        raw_state_loss = (raw_state_loss * target_mask).sum() / target_mask.sum().clamp(min=1.0)

        scaled_state_loss = raw_state_loss * self.semantic_state_loss_weight * self.semantic_state_aux_scale
        denominator = agent_mask.sum().clamp(min=1.0)
        pred_norm = (pred.detach().norm(dim=-1) * agent_mask).sum() / denominator
        target_norm = (target.detach().norm(dim=-1) * agent_mask).sum() / denominator

        state_feature_mean = (target.detach() * target_mask).sum(dim=(0, 1, 2)) / target_mask.sum(dim=(0, 1, 2)).clamp(min=1.0)
        pred_feature_mean = (pred.detach() * target_mask).sum(dim=(0, 1, 2)) / target_mask.sum(dim=(0, 1, 2)).clamp(min=1.0)

        logs = {
            "semantic_state_loss": scaled_state_loss.detach(),
            "semantic_state_raw_l1": raw_state_loss.detach(),
            "semantic_state_target_norm": target_norm.detach(),
            "semantic_state_pred_norm": pred_norm.detach(),
            "semantic_state_aux_scale": pred.new_tensor(self.semantic_state_aux_scale),
        }

        for feature_offset, feature_idx in enumerate(self.semantic_state_feature_indices):
            logs["semantic_state_target_feat_{}".format(feature_idx)] = state_feature_mean[feature_offset].detach()
            logs["semantic_state_pred_feat_{}".format(feature_idx)] = pred_feature_mean[feature_offset].detach()

        return scaled_state_loss, logs

    def _build_attack_action_targets(self, actions):
        actions = actions.squeeze(-1).long()
        attack_targets = actions.new_zeros(
            actions.size(0),
            actions.size(1),
            actions.size(2),
            self.semantic_action_dim,
        ).float()

        attack_mask = actions >= self.semantic_action_offset
        if attack_mask.any():
            attack_indices = (actions[attack_mask] - self.semantic_action_offset).long()
            attack_targets[attack_mask] = F.one_hot(
                attack_indices,
                num_classes=self.semantic_action_dim,
            ).float()

        return attack_targets

    def _build_ally_state_targets(self, obs):
        bs, ts, n_agents, obs_dim = obs.shape
        if obs_dim != self.obs_shape:
            raise ValueError("Unexpected obs dim {} (expected {})".format(obs_dim, self.obs_shape))

        ally_flat = obs[:, :, :, self.ally_feat_start:self.ally_feat_end]
        ally_feats = ally_flat.reshape(bs, ts, n_agents, self.n_agents - 1, self.ally_feat_dim)
        return ally_feats[:, :, :, :, self.semantic_state_feature_indices]

    def _drop_self_attention(self, attn):
        chunks = []
        for agent_idx in range(self.n_agents):
            other_ids = [ally_idx for ally_idx in range(self.n_agents) if ally_idx != agent_idx]
            chunks.append(attn[:, :, agent_idx, other_ids].unsqueeze(2))
        return th.cat(chunks, dim=2)

    def _get_aux_scale(self, t_env, warmup_steps, decay_start, decay_end, final_scale):
        if warmup_steps > 0 and t_env < warmup_steps:
            return min(1.0, float(t_env) / float(warmup_steps))

        if decay_start is None or decay_end is None:
            return 1.0

        if decay_end <= decay_start:
            return final_scale if t_env >= decay_start else 1.0

        if t_env <= decay_start:
            return 1.0
        if t_env >= decay_end:
            return final_scale

        decay_progress = float(t_env - decay_start) / float(decay_end - decay_start)
        return 1.0 + decay_progress * (final_scale - 1.0)

    def _init_obs_layout(self, scheme, args):
        env_args = getattr(args, "env_args", {})
        if env_args.get("obs_pathing_grid", False) or env_args.get("obs_terrain_height", False) or env_args.get("obs_timestep_number", False):
            raise NotImplementedError("semantic_head_v2 currently supports obs without pathing/terrain/timestep features")
        if not env_args.get("obs_all_health", False):
            raise NotImplementedError("semantic_head_v2 requires obs_all_health=True")

        obs_vshape = scheme["obs"]["vshape"]
        self.obs_shape = obs_vshape if isinstance(obs_vshape, int) else obs_vshape[0]
        self.n_enemies = self.args.n_actions - self.semantic_action_offset
        self.move_feats_dim = self.args.env_args.get("n_actions_move", 4)
        last_action_dim = self.args.n_actions if env_args.get("obs_last_action", False) else 0

        numerator = self.obs_shape - self.move_feats_dim + 4 + last_action_dim
        denominator = self.n_enemies + self.n_agents
        if denominator <= 0 or numerator % denominator != 0:
            raise ValueError(
                "Unable to infer ally feature dim from obs_shape={} n_agents={} n_enemies={}".format(
                    self.obs_shape,
                    self.n_agents,
                    self.n_enemies,
                )
            )

        self.ally_feat_dim = numerator // denominator
        self.enemy_feat_dim = self.ally_feat_dim
        self.ally_feat_start = self.move_feats_dim + self.n_enemies * self.enemy_feat_dim
        self.ally_feat_end = self.ally_feat_start + (self.n_agents - 1) * self.ally_feat_dim

        if self.ally_feat_end > self.obs_shape:
            raise ValueError("Inferred ally feature slice exceeds obs dim")
        if max(self.semantic_state_feature_indices) >= self.ally_feat_dim:
            raise ValueError("semantic_state_feature_indices exceed inferred ally feature dim {}".format(self.ally_feat_dim))

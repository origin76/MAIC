import torch as th
import torch.nn.functional as F

from .budgeted_sparse_mappo_learner import BudgetedSparseMAPPOLearner


class BudgetedSparseMAPPOSemanticHeadV1Learner(BudgetedSparseMAPPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super(BudgetedSparseMAPPOSemanticHeadV1Learner, self).__init__(mac, scheme, logger, args)
        self.semantic_action_offset = getattr(args, "semantic_action_offset", 6)
        self.semantic_action_loss_weight = getattr(args, "semantic_action_loss_weight", 0.05)
        self.semantic_aux_warmup_steps = getattr(args, "semantic_aux_warmup_steps", 0)
        self.semantic_aux_decay_start = getattr(args, "semantic_aux_decay_start", None)
        self.semantic_aux_decay_end = getattr(args, "semantic_aux_decay_end", None)
        self.semantic_aux_final_scale = getattr(args, "semantic_aux_final_scale", 1.0)
        self.semantic_aux_scale = 1.0
        self.semantic_action_dim = args.n_actions - self.semantic_action_offset

        if self.semantic_action_dim <= 0:
            raise ValueError("semantic_action_offset must be smaller than n_actions")

    def train(self, batch, t_env, episode_num):
        self.semantic_aux_scale = self._get_semantic_aux_scale(t_env)
        super().train(batch, t_env, episode_num)

    def _forward_policy(self, batch, prepare_for_logging=False):
        outputs = []
        loss_items = []
        logs = []
        semantic_action_preds = []
        semantic_action_attns = []

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

            loss_items.append(extra)

        policy = th.stack(outputs, dim=1)
        merged = self._merge_extra_items(loss_items, logs)
        if len(semantic_action_preds) > 0:
            merged["semantic_action_pred_seq"] = th.stack(semantic_action_preds, dim=1)
        if len(semantic_action_attns) > 0:
            merged["semantic_action_attn_seq"] = th.stack(semantic_action_attns, dim=1)
        return policy, merged

    def _process_extra_losses(self, extra, batch):
        total, loss_dict = super()._process_extra_losses(extra, batch)

        semantic_action_pred = extra.get("semantic_action_pred_seq", None)
        semantic_action_attn = extra.get("semantic_action_attn_seq", None)
        if semantic_action_pred is None or semantic_action_attn is None or self.semantic_action_loss_weight <= 0:
            return total, loss_dict

        transition_mask = batch["filled"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()
        transition_mask[:, 1:] = transition_mask[:, 1:] * (1 - terminated[:, :-1])

        attack_targets = self._build_attack_action_targets(batch["actions"][:, :-1]).detach()
        weighted_attack_targets = th.matmul(semantic_action_attn.detach(), attack_targets)

        target_mask = transition_mask.unsqueeze(-1).expand_as(weighted_attack_targets)
        raw_semantic_loss = F.mse_loss(
            semantic_action_pred,
            weighted_attack_targets,
            reduction="none",
        )
        raw_semantic_loss = (raw_semantic_loss * target_mask).sum() / target_mask.sum().clamp(min=1.0)

        semantic_action_loss = raw_semantic_loss * self.semantic_action_loss_weight * self.semantic_aux_scale
        total = total + semantic_action_loss

        agent_mask = transition_mask.squeeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            weighted_attack_targets.size(2),
        )
        denominator = agent_mask.sum().clamp(min=1.0)
        target_attack_mass = (weighted_attack_targets.detach().sum(dim=-1) * agent_mask).sum() / denominator
        pred_attack_mass = (semantic_action_pred.detach().sum(dim=-1) * agent_mask).sum() / denominator
        positive_ratio = (((weighted_attack_targets.detach().sum(dim=-1) > 0).float()) * agent_mask).sum() / denominator

        loss_dict["semantic_action_loss"] = semantic_action_loss.detach()
        loss_dict["semantic_action_raw_mse"] = raw_semantic_loss.detach()
        loss_dict["semantic_target_attack_mass"] = target_attack_mass.detach()
        loss_dict["semantic_pred_attack_mass"] = pred_attack_mass.detach()
        loss_dict["semantic_positive_ratio"] = positive_ratio.detach()
        loss_dict["semantic_aux_scale"] = semantic_action_pred.new_tensor(self.semantic_aux_scale)

        return total, loss_dict

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

    def _get_semantic_aux_scale(self, t_env):
        if self.semantic_aux_warmup_steps > 0 and t_env < self.semantic_aux_warmup_steps:
            return min(1.0, float(t_env) / float(self.semantic_aux_warmup_steps))

        if self.semantic_aux_decay_start is None or self.semantic_aux_decay_end is None:
            return 1.0

        if self.semantic_aux_decay_end <= self.semantic_aux_decay_start:
            return self.semantic_aux_final_scale if t_env >= self.semantic_aux_decay_start else 1.0

        if t_env <= self.semantic_aux_decay_start:
            return 1.0
        if t_env >= self.semantic_aux_decay_end:
            return self.semantic_aux_final_scale

        decay_progress = (
            float(t_env - self.semantic_aux_decay_start) /
            float(self.semantic_aux_decay_end - self.semantic_aux_decay_start)
        )
        return 1.0 + decay_progress * (self.semantic_aux_final_scale - 1.0)

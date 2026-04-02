import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class MAICV1Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        prepare_for_logging = True if t_env - self.log_stats_t >= self.args.learner_log_interval else False
        aux_scale = self._get_aux_scale(t_env)

        logs = []
        losses = []

        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs, returns_ = self.mac.forward(
                batch,
                t=t,
                prepare_for_logging=prepare_for_logging,
                train_mode=True,
                mixer=self.target_mixer,
            )

            if t < rewards.size(1):
                returns_ = self._add_auxiliary_losses(
                    returns_,
                    actions[:, t],
                    rewards[:, t],
                    mask[:, t],
                    aux_scale,
                    prepare_for_logging,
                )

            mac_out.append(agent_outs)
            if prepare_for_logging and "logs" in returns_:
                logs.append(returns_["logs"])
                del returns_["logs"]
            losses.append(returns_)

        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("aux_loss_scale", aux_scale, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )

            self._log_for_scalar_and_histogram(logs, t_env)
            self._log_for_loss(loss_dict, t_env)
            self.log_stats_t = t_env

    def _add_auxiliary_losses(self, returns_, action_targets, reward_targets, transition_mask, aux_scale, prepare_for_logging):
        logs = returns_.get("logs", {})
        mask_scalar = transition_mask.squeeze(-1)

        intent_logits = returns_.pop("intent_logits", None)
        if intent_logits is not None:
            bs, n_agents, _, n_actions = intent_logits.shape
            target_actions = action_targets.squeeze(-1).long().unsqueeze(2).expand(-1, -1, n_agents)
            self_mask = th.eye(n_agents, device=intent_logits.device, dtype=th.bool).unsqueeze(0)
            valid_pairs = (~self_mask).float() * mask_scalar.view(bs, 1, 1)

            intent_ce = F.cross_entropy(
                intent_logits.reshape(-1, n_actions),
                target_actions.reshape(-1),
                reduction="none",
            ).view(bs, n_agents, n_agents)
            intent_denominator = valid_pairs.sum().clamp(min=1.0)
            raw_intent_loss = (intent_ce * valid_pairs).sum() / intent_denominator
            returns_["intent_aux_loss"] = raw_intent_loss * self.args.intent_aux_loss_weight * aux_scale

            if prepare_for_logging:
                predicted_actions = intent_logits.argmax(dim=-1)
                intent_acc = ((predicted_actions == target_actions).float() * valid_pairs).sum() / intent_denominator
                logs["Scalar_intent_aux_acc"] = intent_acc
                logs["Scalar_intent_aux_ce"] = raw_intent_loss.detach()

        reward_pred = returns_.pop("reward_pred", None)
        if reward_pred is not None:
            reward_weight = mask_scalar.unsqueeze(-1)
            reward_denominator = reward_weight.sum().clamp(min=1.0)
            raw_reward_loss = (((reward_pred - reward_targets) ** 2) * reward_weight).sum() / reward_denominator
            returns_["reward_aux_loss"] = raw_reward_loss * self.args.reward_aux_loss_weight * aux_scale

            if prepare_for_logging:
                logs["Scalar_reward_aux_mse"] = raw_reward_loss.detach()
                logs["Scalar_reward_pred_mean"] = reward_pred.detach().mean()

        if prepare_for_logging:
            logs["Scalar_aux_scale"] = th.tensor(aux_scale, device=transition_mask.device)
            returns_["logs"] = logs
        elif "logs" in returns_:
            returns_["logs"] = logs

        return returns_

    def _get_aux_scale(self, t_env):
        warmup_steps = max(0, getattr(self.args, "aux_loss_warmup_steps", 0))
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(t_env) / float(warmup_steps))

    def _log_for_scalar_and_histogram(self, logs, t):
        if len(logs) == 0:
            return

        keys = list(logs[0].keys())
        for k in keys:
            log_key = "_".join(k.split("_")[1:])
            if str(k).startswith("Histogram"):
                value = th.stack([l[k] for l in logs], dim=1)
                self.logger.log_histogram(log_key, value, t)
            elif str(k).startswith("Scalar"):
                values = [l[k].float() if isinstance(l[k], th.Tensor) else th.tensor(l[k]) for l in logs]
                self.logger.log_stat(log_key, th.stack(values).mean(), t)

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith("loss"):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict.keys():
            loss_dict[k] /= batch.max_seq_length
        total_loss /= batch.max_seq_length
        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for k, v in losses.items():
            self.logger.log_stat(k, v.item(), t)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

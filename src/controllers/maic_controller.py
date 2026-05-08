from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class MAICMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.eval_peer_local_diagnostic = bool(
            getattr(args, "eval_peer_local_diagnostic", False)
        )
        self._eval_peer_local_diag = {}

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        collect_sequence_data = test_mode and self.eval_peer_local_diagnostic
        agent_outputs, diag_extra = self.forward(
            ep_batch,
            t_ep,
            test_mode=test_mode,
            train_mode=False,
            t_env=t_env,
            collect_sequence_data=collect_sequence_data,
        )
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if collect_sequence_data:
            self._accumulate_eval_peer_local_diagnostic(diag_extra, chosen_actions, bs)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, losses = self.agent.forward(
            agent_inputs,
            self.hidden_states,
            ep_batch.batch_size,
            test_mode=test_mode,
            avail_actions=avail_actions,
            raw_obs=ep_batch["obs"][:, t],
            **kwargs
        )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), losses

    def reset_eval_peer_local_diagnostic(self):
        self._eval_peer_local_diag = {}

    def pop_eval_peer_local_diagnostic(self):
        diag = self._eval_peer_local_diag
        self._eval_peer_local_diag = {}
        return diag

    def _accumulate_eval_peer_local_diagnostic(self, extra, chosen_actions, bs):
        if not isinstance(extra, dict):
            return

        required_keys = (
            "seq_counterfactual_attack_local_top1",
            "seq_counterfactual_attack_peer_top1",
            "seq_counterfactual_attack_fused_top1",
            "seq_counterfactual_attack_attack_only_top1",
            "seq_counterfactual_attack_local_top1_prob",
            "seq_counterfactual_attack_peer_top1_prob",
            "seq_counterfactual_attack_fused_top1_prob",
            "seq_counterfactual_attack_attack_only_top1_prob",
            "seq_counterfactual_attack_peer_valid_mask",
            "seq_counterfactual_attack_can_mask",
            "seq_counterfactual_attack_no_comm_prob",
            "seq_counterfactual_attack_gate",
            "seq_counterfactual_attack_delta_abs",
            "seq_counterfactual_attack_effective_delta_abs",
        )
        if any(key not in extra for key in required_keys):
            return

        def _slice(key):
            value = extra[key]
            if not th.is_tensor(value):
                return None
            return value[bs].detach()

        local_top1 = _slice("seq_counterfactual_attack_local_top1").long()
        peer_top1 = _slice("seq_counterfactual_attack_peer_top1").long()
        fused_top1 = _slice("seq_counterfactual_attack_fused_top1").long()
        attack_only_top1 = _slice("seq_counterfactual_attack_attack_only_top1").long()
        local_top1_prob = _slice("seq_counterfactual_attack_local_top1_prob").float()
        peer_top1_prob = _slice("seq_counterfactual_attack_peer_top1_prob").float()
        fused_top1_prob = _slice("seq_counterfactual_attack_fused_top1_prob").float()
        attack_only_top1_prob = _slice("seq_counterfactual_attack_attack_only_top1_prob").float()
        peer_valid = _slice("seq_counterfactual_attack_peer_valid_mask").float()
        attack_can = _slice("seq_counterfactual_attack_can_mask").float()
        no_comm_prob = _slice("seq_counterfactual_attack_no_comm_prob").float()
        attack_gate = _slice("seq_counterfactual_attack_gate").float()
        attack_delta_abs = _slice("seq_counterfactual_attack_delta_abs").float()
        effective_delta_abs = _slice(
            "seq_counterfactual_attack_effective_delta_abs"
        ).float()

        valid = attack_can * peer_valid
        valid_count = valid.sum()
        conflict = (peer_top1 != local_top1).float() * valid
        conflict_count = conflict.sum()
        chosen_actions = chosen_actions.detach()
        if chosen_actions.dim() == 3 and chosen_actions.size(-1) == 1:
            chosen_actions = chosen_actions.squeeze(-1)
        semantic_action_offset = int(getattr(self.args, "semantic_action_offset", 6))
        chosen_attack = (chosen_actions >= semantic_action_offset).float() * valid
        chosen_attack_count = chosen_attack.sum()
        chosen_attack_target = (
            chosen_actions - semantic_action_offset
        ).clamp(min=0).long()
        chosen_attack_match_local = (
            (chosen_attack_target == local_top1).float() * chosen_attack
        )
        chosen_attack_match_peer = (
            (chosen_attack_target == peer_top1).float() * chosen_attack
        )
        chosen_attack_match_fused = (
            (chosen_attack_target == fused_top1).float() * chosen_attack
        )
        chosen_attack_match_attack_only = (
            (chosen_attack_target == attack_only_top1).float() * chosen_attack
        )

        def add_sum(name, value):
            self._eval_peer_local_diag[name] = self._eval_peer_local_diag.get(name, 0.0) + float(
                value.detach().sum().item()
            )

        def add_ratio(name, numerator, denominator):
            add_sum(name + "_sum", numerator)
            add_sum(name + "_denom", denominator)

        add_sum("peer_local_attack_valid_count", valid_count)
        add_sum("peer_local_conflict_count", conflict_count)
        add_sum("peer_local_chosen_attack_count", chosen_attack_count)
        add_ratio("peer_local_conflict_rate", conflict, valid_count)
        add_ratio(
            "peer_local_local_agreement_rate",
            (peer_top1 == local_top1).float() * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_fused_follow_peer_on_conflict_rate",
            (fused_top1 == peer_top1).float() * conflict,
            conflict_count,
        )
        add_ratio(
            "peer_local_fused_stay_local_on_conflict_rate",
            (fused_top1 == local_top1).float() * conflict,
            conflict_count,
        )
        add_ratio(
            "peer_local_fused_flip_rate",
            (fused_top1 != local_top1).float() * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_attack_only_flip_rate",
            (attack_only_top1 != local_top1).float() * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_attack_only_follow_peer_on_conflict_rate",
            (attack_only_top1 == peer_top1).float() * conflict,
            conflict_count,
        )
        add_ratio(
            "peer_local_attack_only_stay_local_on_conflict_rate",
            (attack_only_top1 == local_top1).float() * conflict,
            conflict_count,
        )
        add_ratio(
            "peer_local_attack_only_other_on_conflict_rate",
            (
                (attack_only_top1 != peer_top1).float()
                * (attack_only_top1 != local_top1).float()
                * conflict
            ),
            conflict_count,
        )
        add_ratio(
            "peer_local_fused_other_on_conflict_rate",
            (
                (fused_top1 != peer_top1).float()
                * (fused_top1 != local_top1).float()
                * conflict
            ),
            conflict_count,
        )
        add_ratio(
            "peer_local_chosen_attack_rate",
            chosen_attack,
            valid_count,
        )
        add_ratio(
            "peer_local_chosen_match_local_rate",
            chosen_attack_match_local,
            chosen_attack_count,
        )
        add_ratio(
            "peer_local_chosen_match_peer_rate",
            chosen_attack_match_peer,
            chosen_attack_count,
        )
        add_ratio(
            "peer_local_chosen_match_fused_rate",
            chosen_attack_match_fused,
            chosen_attack_count,
        )
        add_ratio(
            "peer_local_chosen_match_attack_only_rate",
            chosen_attack_match_attack_only,
            chosen_attack_count,
        )
        add_ratio(
            "peer_local_local_top1_prob_mean",
            local_top1_prob * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_peer_top1_prob_mean",
            peer_top1_prob * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_fused_top1_prob_mean",
            fused_top1_prob * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_attack_only_top1_prob_mean",
            attack_only_top1_prob * valid,
            valid_count,
        )
        add_ratio(
            "peer_local_no_comm_prob_mean",
            no_comm_prob * attack_can,
            attack_can.sum(),
        )
        add_ratio(
            "peer_local_gate_mean",
            attack_gate * attack_can,
            attack_can.sum(),
        )
        add_ratio(
            "peer_local_delta_abs_mean",
            attack_delta_abs * attack_can,
            attack_can.sum(),
        )
        add_ratio(
            "peer_local_effective_delta_abs_mean",
            effective_delta_abs * attack_can,
            attack_can.sum(),
        )
        add_ratio(
            "peer_local_peer_valid_rate",
            peer_valid,
            attack_can,
        )
        add_ratio(
            "peer_local_peer_conflict_rate",
            conflict,
            attack_can,
        )

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path, strict=True):
        state_dict = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        return self.agent.load_state_dict(state_dict, strict=strict)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

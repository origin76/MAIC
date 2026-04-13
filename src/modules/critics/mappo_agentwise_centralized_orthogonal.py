import torch as th
import torch.nn as nn
import torch.nn.functional as F


def init_linear(layer, gain=1.0, bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)


class MAPPOAgentWiseCentralizedOrthogonalCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MAPPOAgentWiseCentralizedOrthogonalCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = getattr(args, "critic_hidden_dim", 128)

        self.use_obs = getattr(args, "critic_use_local_obs", True)
        self.use_last_action = getattr(args, "critic_use_last_action", True)
        self.use_agent_id = getattr(args, "critic_use_agent_id", True)

        input_shape = self._get_input_shape(scheme)

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

        if getattr(args, "use_orthogonal", True):
            self._reset_parameters()

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []

        state = batch["state"][:, ts].reshape(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        inputs.append(state)

        if self.use_obs:
            inputs.append(batch["obs"][:, ts].reshape(bs, max_t, self.n_agents, -1))

        if self.use_last_action:
            if t == 0:
                last_actions = th.zeros_like(batch["actions_onehot"][:, 0:1])
            elif isinstance(t, int):
                last_actions = batch["actions_onehot"][:, slice(t - 1, t)]
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                    dim=1,
                )
            last_actions = last_actions.reshape(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        if self.use_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0)
            agent_ids = agent_ids.expand(bs, max_t, -1, -1)
            inputs.append(agent_ids)

        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = self._vshape_dim(scheme["state"]["vshape"])

        if self.use_obs:
            input_shape += self._vshape_dim(scheme["obs"]["vshape"])

        if self.use_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        if self.use_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _vshape_dim(self, vshape):
        if isinstance(vshape, int):
            return vshape

        dim = 1
        for item in vshape:
            dim *= item
        return dim

    def _reset_parameters(self):
        relu_gain = nn.init.calculate_gain("relu")
        output_gain = getattr(self.args, "critic_output_gain", 1.0)

        init_linear(self.fc1, gain=relu_gain)
        init_linear(self.fc2, gain=relu_gain)
        init_linear(self.fc3, gain=output_gain)

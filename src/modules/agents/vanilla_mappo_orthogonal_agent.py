import torch.nn as nn
import torch.nn.functional as F


def init_linear(layer, gain=1.0, bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)


def init_gru_cell(gru):
    for name, param in gru.named_parameters():
        if "weight" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class VanillaMAPPOOrthogonalAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOOrthogonalAgent, self).__init__()
        self.args = args

        hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.n_actions),
        )

        if getattr(args, "use_orthogonal", True):
            self._reset_parameters()

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        logits = self.policy_head(h)
        return logits, h, {}

    def _reset_parameters(self):
        relu_gain = nn.init.calculate_gain("relu")
        output_gain = getattr(self.args, "actor_output_gain", 0.01)

        init_linear(self.fc1, gain=relu_gain)
        init_gru_cell(self.rnn)
        init_linear(self.policy_head[0], gain=relu_gain)
        init_linear(self.policy_head[2], gain=output_gain)

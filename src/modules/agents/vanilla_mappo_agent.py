import torch.nn as nn
import torch.nn.functional as F


class VanillaMAPPOAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VanillaMAPPOAgent, self).__init__()
        self.args = args

        hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.n_actions),
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        logits = self.policy_head(h)
        return logits, h, {}

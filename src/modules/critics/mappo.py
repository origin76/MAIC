import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAPPOCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MAPPOCritic, self).__init__()
        self.args = args
        self.hidden_dim = getattr(args, "critic_hidden_dim", 128)

        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, tuple):
            input_dim = 1
            for item in input_shape:
                input_dim *= item
        else:
            input_dim = input_shape

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, batch, t=None):
        states = batch["state"] if t is None else batch["state"][:, t:t + 1]
        x = states.reshape(states.size(0), states.size(1), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

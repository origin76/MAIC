import torch as th

from .budgeted_sparse_mappo_agent import BudgetedSparseMAPPOAgent


class BudgetedSparseMAPPONoEpsBias1Agent(BudgetedSparseMAPPOAgent):
    def __init__(self, input_shape, args):
        super(BudgetedSparseMAPPONoEpsBias1Agent, self).__init__(input_shape, args)
        # sigmoid(1.0) ~= 0.73, which keeps communication heads open early
        # without immediately saturating them at 1.
        th.nn.init.constant_(self.head_gate.bias, 1.0)

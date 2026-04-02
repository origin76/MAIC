REGISTRY = {}

from .rnn_agent import RNNAgent
from .maic_agent import MAICAgent
from .maic_multihead_agent import MAICMultiHeadAgent
from .maic_v1_agent import MAICV1Agent
from .budgeted_sparse_mappo_agent import BudgetedSparseMAPPOAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY['maic'] = MAICAgent
REGISTRY["maic_multihead"] = MAICMultiHeadAgent
REGISTRY["maic_v1"] = MAICV1Agent
REGISTRY["budgeted_sparse_mappo"] = BudgetedSparseMAPPOAgent

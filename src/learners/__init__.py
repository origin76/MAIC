from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .maic_learner import MAICLearner
from .maic_qplex_learner import MAICQPLEXLearner
from .maic_multihead_learner import MAICMultiHeadLearner
from .maic_v1_learner import MAICV1Learner
from .budgeted_sparse_mappo_learner import BudgetedSparseMAPPOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY['maic_learner'] = MAICLearner
REGISTRY['maic_qplex_learner'] = MAICQPLEXLearner
REGISTRY["maic_multihead_learner"] = MAICMultiHeadLearner
REGISTRY["maic_v1_learner"] = MAICV1Learner
REGISTRY["budgeted_sparse_mappo_learner"] = BudgetedSparseMAPPOLearner

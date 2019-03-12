from garage.tf.q_functions.base import QFunction
from garage.tf.q_functions.base2 import QFunction2
from garage.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction)
from garage.tf.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction

__all__ = [
    "QFunction", "QFunction2", "ContinuousMLPQFunction",
    "DiscreteCNNQFunction", "DiscreteMLPQFunction"
]

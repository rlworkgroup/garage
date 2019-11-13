"""Q-Functions for TensorFlow-based algorithms."""
from garage.tf.q_functions.base import QFunction
from garage.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction)
from garage.tf.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction

__all__ = [
    'QFunction', 'ContinuousMLPQFunction', 'DiscreteCNNQFunction',
    'DiscreteMLPQFunction'
]

"""Policies for TensorFlow-based algorithms."""
from garage.tf.policies.base import Policy
from garage.tf.policies.base import StochasticPolicy
from garage.tf.policies.base import StochasticPolicy2
from garage.tf.policies.categorical_cnn_policy import CategoricalCNNPolicy
from garage.tf.policies.categorical_cnn_policy2 import CategoricalCNNPolicy2
from garage.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from garage.tf.policies.categorical_gru_policy2 import CategoricalGRUPolicy2
from garage.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from garage.tf.policies.categorical_lstm_policy2 import CategoricalLSTMPolicy2
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from garage.tf.policies.categorical_mlp_policy2 import CategoricalMLPPolicy2
from garage.tf.policies.continuous_mlp_policy import ContinuousMLPPolicy
from garage.tf.policies.discrete_qf_derived_policy import (
    DiscreteQfDerivedPolicy)
from garage.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from garage.tf.policies.gaussian_gru_policy2 import GaussianGRUPolicy2
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.gaussian_lstm_policy2 import GaussianLSTMPolicy2
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_mlp_policy2 import GaussianMLPPolicy2
from garage.tf.policies.gaussian_mlp_task_embedding_policy import (
    GaussianMLPTaskEmbeddingPolicy)
from garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy

__all__ = [
    'Policy', 'StochasticPolicy', 'StochasticPolicy2', 'CategoricalCNNPolicy',
    'CategoricalCNNPolicy2', 'CategoricalGRUPolicy', 'CategoricalGRUPolicy2',
    'CategoricalLSTMPolicy', 'CategoricalLSTMPolicy2', 'CategoricalMLPPolicy',
    'CategoricalMLPPolicy2', 'ContinuousMLPPolicy', 'DiscreteQfDerivedPolicy',
    'GaussianGRUPolicy', 'GaussianGRUPolicy2', 'GaussianLSTMPolicy',
    'GaussianLSTMPolicy2', 'GaussianMLPPolicy', 'GaussianMLPPolicy2',
    'GaussianMLPTaskEmbeddingPolicy'
]

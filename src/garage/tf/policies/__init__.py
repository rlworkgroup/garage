"""Policies for TensorFlow-based algorithms."""
from garage.tf.policies.categorical_cnn_policy import CategoricalCNNPolicy
from garage.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from garage.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from garage.tf.policies.continuous_mlp_policy import ContinuousMLPPolicy
from garage.tf.policies.discrete_qf_derived_policy import (
    DiscreteQfDerivedPolicy)
from garage.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_mlp_task_embedding_policy import (
    GaussianMLPTaskEmbeddingPolicy)
from garage.tf.policies.policy import Policy
from garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy

__all__ = [
    'Policy', 'CategoricalCNNPolicy', 'CategoricalGRUPolicy',
    'CategoricalLSTMPolicy', 'CategoricalMLPPolicy', 'ContinuousMLPPolicy',
    'DiscreteQfDerivedPolicy', 'GaussianGRUPolicy', 'GaussianLSTMPolicy',
    'GaussianMLPPolicy', 'GaussianMLPTaskEmbeddingPolicy',
    'TaskEmbeddingPolicy'
]

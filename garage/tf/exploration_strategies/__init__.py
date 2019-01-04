"""
Exploration strategies used for RL algorithms.
"""
from garage.tf.exploration_strategies.epsilon_greedy_strategy import (
    EpsilonGreedyStrategy)
from garage.tf.exploration_strategies.ou_strategy import OUStrategy

__all__ = ["OUStrategy", "EpsilonGreedyStrategy"]

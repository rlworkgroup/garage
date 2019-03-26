"""Exploration strategies which use NumPy as a numerical backend."""
from garage.np.exploration_strategies.base import ExplorationStrategy
from garage.np.exploration_strategies.epsilon_greedy_strategy import (
    EpsilonGreedyStrategy)
from garage.np.exploration_strategies.ou_strategy import OUStrategy

__all__ = ['EpsilonGreedyStrategy', 'ExplorationStrategy', 'OUStrategy']

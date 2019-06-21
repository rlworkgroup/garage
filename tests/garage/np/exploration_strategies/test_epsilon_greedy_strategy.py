"""Tests for epsilon greedy strategy."""
import pickle

import numpy as np

from garage.np.exploration_strategies import EpsilonGreedyStrategy
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class SimplePolicy:
    """Simple policy for testing."""

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def get_action(self, observation):
        return self.env_spec.action_space.sample()

    def get_actions(self, observations):
        return np.full(len(observations), self.env_spec.action_space.sample())


class TestEpsilonGreedyStrategy:
    def setup_method(self):
        self.env = DummyDiscreteEnv()
        self.policy = SimplePolicy(env_spec=self.env)
        self.epsilon_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=self.env,
            total_timesteps=100,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        self.env.reset()

    def test_epsilon_greedy_strategy(self):
        obs, _, _, _ = self.env.step(1)

        action, _ = self.epsilon_greedy_strategy.get_action(
            0, obs, self.policy)
        assert self.env.action_space.contains(action)

        # epsilon decay by 1 step, new epsilon = 1 - 0.98 = 0.902
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_strategy._epsilon
        assert np.isclose([0.902], [sum(random_rate) / 100000], atol=0.01)

        actions, _ = self.epsilon_greedy_strategy.get_actions(
            0, [obs] * 5, self.policy)

        # epsilon decay by 6 steps in total, new epsilon = 1 - 6 * 0.98 = 0.412
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_strategy._epsilon
        assert np.isclose([0.412], [sum(random_rate) / 100000], atol=0.01)

        for action in actions:
            assert self.env.action_space.contains(action)

    def test_epsilon_greedy_strategy_is_pickleable(self):
        obs, _, _, _ = self.env.step(1)
        for _ in range(5):
            self.epsilon_greedy_strategy.get_action(0, obs, self.policy)

        h_data = pickle.dumps(self.epsilon_greedy_strategy)
        strategy = pickle.loads(h_data)
        assert strategy._epsilon == self.epsilon_greedy_strategy._epsilon

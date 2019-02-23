"""Tests for epsilon greedy strategy."""
import numpy as np
import unittest

from garage.exploration_strategies import EpsilonGreedyStrategy
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class SimplePolicy:
    def __init__(self, env_spec):
        self.env_spec = env_spec

    def get_action(self, observation):
        return self.env_spec.action_space.sample()

    def get_actions(self, observations):
        return np.full(len(observations), self.env_spec.action_space.sample())


class TestEpsilonGreedyStrategy(unittest.TestCase):
    def test_epsilon_greedy_strategy(self):
        env = DummyDiscreteEnv()
        policy = SimplePolicy(env_spec=env)

        # decay from 1.0 to 0.02 within 100 * 0.1 = 10 steps
        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env,
            total_step=100,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        env.reset()
        obs, _, _, _ = env.step(1)

        action = epilson_greedy_strategy.get_action(0, obs, policy)
        assert env.action_space.contains(action)

        # epsilon decay by 1 step, new epsilon = 1 - 0.98 = 0.902
        random_rate = np.random.random(
            100000) < epilson_greedy_strategy._epsilon
        print(sum(random_rate) / 100000)
        assert np.isclose([0.902], [sum(random_rate) / 100000], atol=0.01)

        actions = epilson_greedy_strategy.get_actions(0, [obs] * 5, policy)

        # epsilon decay by 6 steps in total, new epsilon = 1 - 6 * 0.98 = 0.412
        random_rate = np.random.random(
            100000) < epilson_greedy_strategy._epsilon
        print(sum(random_rate) / 100000)
        assert np.isclose([0.412], [sum(random_rate) / 100000], atol=0.01)

        for action in actions:
            assert env.action_space.contains(action)

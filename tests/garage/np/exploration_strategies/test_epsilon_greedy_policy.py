"""Tests for epsilon greedy policy."""
import pickle

import numpy as np

from garage.np.exploration_policies import EpsilonGreedyPolicy

from tests.fixtures.envs.dummy import DummyDiscreteEnv


class SimplePolicy:
    """Simple policy for testing."""

    def __init__(self, env_spec):
        self.env_spec = env_spec

    def get_action(self, _):
        return self.env_spec.action_space.sample(), dict()

    def get_actions(self, observations):
        return np.full(len(observations),
                       self.env_spec.action_space.sample()), dict()


class TestEpsilonGreedyPolicy:

    def setup_method(self):
        self.env = DummyDiscreteEnv()
        self.policy = SimplePolicy(env_spec=self.env)
        self.epsilon_greedy_policy = EpsilonGreedyPolicy(env_spec=self.env,
                                                         policy=self.policy,
                                                         total_timesteps=100,
                                                         max_epsilon=1.0,
                                                         min_epsilon=0.02,
                                                         decay_ratio=0.1)

        self.env.reset()

    def test_epsilon_greedy_policy(self):
        obs, _, _, _ = self.env.step(1)

        action, _ = self.epsilon_greedy_policy.get_action(obs)
        assert self.env.action_space.contains(action)

        # epsilon decay by 1 step, new epsilon = 1 - 0.98 = 0.902
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_policy._epsilon
        assert np.isclose([0.902], [sum(random_rate) / 100000], atol=0.01)

        actions, _ = self.epsilon_greedy_policy.get_actions([obs] * 5)

        # epsilon decay by 6 steps in total, new epsilon = 1 - 6 * 0.98 = 0.412
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_policy._epsilon
        assert np.isclose([0.412], [sum(random_rate) / 100000], atol=0.01)

        for action in actions:
            assert self.env.action_space.contains(action)

    def test_epsilon_greedy_policy_is_pickleable(self):
        obs, _, _, _ = self.env.step(1)
        for _ in range(5):
            self.epsilon_greedy_policy.get_action(obs)

        h_data = pickle.dumps(self.epsilon_greedy_policy)
        policy = pickle.loads(h_data)
        assert policy._epsilon == self.epsilon_greedy_policy._epsilon

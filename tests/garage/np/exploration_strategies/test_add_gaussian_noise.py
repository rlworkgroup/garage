"""Tests for epsilon greedy policy."""
import collections

import numpy as np
import pytest

from garage.np.exploration_policies import AddGaussianNoise

from tests.fixtures.envs.dummy import DummyBoxEnv


@pytest.fixture
def env():
    return DummyBoxEnv()


class ConstantPolicy:
    """Simple policy for testing."""

    def __init__(self, action):
        self.action = action

    def get_action(self, _):
        return self.action, dict()

    def get_actions(self, observations):
        return np.full(len(observations), self.action), dict()

    def reset(self, *args, **kwargs):
        pass

    def get_param_values(self):
        return {'action': self.action}

    def set_param_values(self, params):
        self.action = params['action']


def test_params(env):
    policy1 = ConstantPolicy(env.action_space.sample())
    policy2 = ConstantPolicy(env.action_space.sample())
    assert (policy1.get_action(None)[0] != policy2.get_action(None)[0]).all()

    exp_policy1 = AddGaussianNoise(env, policy1, 1)
    exp_policy2 = AddGaussianNoise(env, policy2, 1)
    exp_policy2.get_action(None)

    assert exp_policy1._sigma() != exp_policy2._sigma()

    exp_policy1.set_param_values(exp_policy2.get_param_values())

    assert (policy1.get_action(None)[0] == policy2.get_action(None)[0]).all()
    assert exp_policy1._sigma() == exp_policy2._sigma()


def test_decay_period(env):
    policy = ConstantPolicy(env.action_space.sample())
    exp_policy = AddGaussianNoise(env,
                                  policy,
                                  total_timesteps=2,
                                  max_sigma=1.,
                                  min_sigma=0.)
    assert (exp_policy.get_action(None)[0] != policy.get_action(None)[0]).all()
    assert (exp_policy.get_action(None)[0] != policy.get_action(None)[0]).all()
    assert (exp_policy.get_action(None)[0] == policy.get_action(None)[0]).all()


def test_update(env):
    policy = ConstantPolicy(env.action_space.sample())
    exp_policy = AddGaussianNoise(env,
                                  policy,
                                  total_timesteps=10,
                                  max_sigma=1.,
                                  min_sigma=0.)
    exp_policy.get_action(None)
    exp_policy.get_action(None)

    DummyBatch = collections.namedtuple('EpisodeBatch', ['lengths'])
    batch = DummyBatch(np.array([1, 2]))

    # new sigma will be 1 - 0.1 * (1 + 2) = 0.7
    exp_policy.update(batch)
    assert np.isclose(exp_policy._sigma(), 0.7)

    exp_policy.get_action(None)

    batch = DummyBatch(np.array([1, 1, 2]))
    # new sigma will be 0.7 - 0.1 * (1 + 1 + 2) = 0.3
    exp_policy.update(batch)
    assert np.isclose(exp_policy._sigma(), 0.3)

"""Tests for epsilon greedy policy."""
import pickle

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

    exp_policy1 = AddGaussianNoise(env, policy1)
    exp_policy2 = AddGaussianNoise(env, policy2)
    exp_policy1.set_param_values(exp_policy2.get_param_values())

    assert (policy1.get_action(None)[0] == policy2.get_action(None)[0]).all()


def test_decay_period(env):
    policy = ConstantPolicy(env.action_space.sample())
    exp_policy = AddGaussianNoise(env,
                                  policy,
                                  max_sigma=1.,
                                  min_sigma=0.,
                                  decay_period=2)
    assert (exp_policy.get_action(None)[0] != policy.get_action(None)[0]).all()
    exp_policy.reset()
    assert (exp_policy.get_action(None)[0] != policy.get_action(None)[0]).all()
    exp_policy.reset()
    assert (exp_policy.get_action(None)[0] == policy.get_action(None)[0]).all()

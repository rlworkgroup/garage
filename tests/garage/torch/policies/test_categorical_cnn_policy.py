"""Test categoricalCNNPolicy."""
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GarageEnv
from garage.torch.policies import CategoricalCNNPolicy
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestCategoricalCNNPolicies:

    @pytest.mark.parametrize(
        'hidden_channels, kernel_sizes, strides, hidden_sizes', [
            ((3, ), (3, ), (1, ), (4, )),
            ((3, 3), (3, 3), (1, 1), (4, 4)),
            ((3, 3), (3, 3), (2, 2), (4, 4)),
        ])
    def test_get_action(self, hidden_channels, kernel_sizes, strides,
                        hidden_sizes):
        """Test get_action function."""
        env = GarageEnv(DummyDiscretePixelEnv())
        # obs_dim = env_spec.observation_space.flat_dim
        # act_dim = env_spec.action_space.flat_dim
        # obs = torch.ones(obs_dim, dtype=torch.float32)
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      kernel_sizes=kernel_sizes,
                                      hidden_channels=hidden_channels,
                                      strides=strides,
                                      hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)
        action, prob = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)

    def test_get_action_np(self):
        pass

    def test_get_actions(self):
        pass

    def test_get_actions_np(self):
        pass

    def test_is_pickleable(self):
        pass

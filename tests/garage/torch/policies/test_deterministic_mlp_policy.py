import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.tf.envs import TfEnv
from garage.torch.policies import DeterministicMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestDeterministicMLPPolicies:
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 1), (2, 2)])
    # yapf: enable
    def test_get_action(self, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([1, obs_dim], dtype=torch.float32)

        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=hidden_sizes,
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)

        expected_output = np.full([1, act_dim],
                                  fill_value=obs_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(policy.get_action(obs)[0], expected_output)

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (4, (2, )),
        (6, (3, )),
        (20, (1, 1)),
        (32, (2, 6, 8)),
    ])
    # yapf: enable
    def test_get_actions(self, batch_size, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)

        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=hidden_sizes,
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)

        expected_output = np.full([batch_size, act_dim],
                                  fill_value=obs_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(policy.get_actions(obs)[0], expected_output)

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (4, (2, )),
        (10, (3, )),
        (25, (2, 4)),
        (34, (2, 6, 11)),
    ])
    # yapf: enable
    def test_is_pickleable(self, batch_size, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)

        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=hidden_sizes,
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)

        output1 = policy.get_actions(obs)[0]

        p = pickle.dumps(policy)
        policy_pickled = pickle.loads(p)
        output2 = policy_pickled.get_actions(obs)[0]
        assert np.array_equal(output1, output2)

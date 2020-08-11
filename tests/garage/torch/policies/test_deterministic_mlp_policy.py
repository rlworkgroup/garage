import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GymEnv
from garage.torch.policies import DeterministicMLPPolicy

# yapf: Disable
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDictEnv

# yapf: Enable


class TestDeterministicMLPPolicies:
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 1), (2, 2)])
    # yapf: enable
    def test_get_action(self, hidden_sizes):
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([1, obs_dim], dtype=torch.float32)
        obs_np = np.ones([1, obs_dim], dtype=np.float32)
        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=hidden_sizes,
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)

        expected_output = np.full([1, act_dim],
                                  fill_value=obs_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(policy.get_action(obs)[0], expected_output)
        assert np.array_equal(policy.get_action(obs_np)[0], expected_output)

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
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
        obs_np = np.ones([obs_dim], dtype=np.float32)
        obs_torch = torch.Tensor(obs_np)
        policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=hidden_sizes,
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)

        expected_output = np.full([batch_size, act_dim],
                                  fill_value=obs_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(policy.get_actions(obs)[0], expected_output)
        assert np.array_equal(
            policy.get_actions([obs_torch] * batch_size)[0], expected_output)
        assert np.array_equal(
            policy.get_actions([obs_np] * batch_size)[0], expected_output)

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
        env_spec = GymEnv(DummyBoxEnv())
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

    def test_get_action_dict_space(self):
        """Test if observations from dict obs spaces are properly flattened."""
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_nonlinearity=None,
                                        hidden_sizes=(1, ),
                                        hidden_w_init=nn.init.ones_,
                                        output_w_init=nn.init.ones_)
        obs = env.reset()[0]

        action, _ = policy.get_action(obs)
        assert env.action_space.shape == action.shape

        actions, _ = policy.get_actions(np.array([obs, obs]))
        for action in actions:
            assert env.action_space.shape == action.shape

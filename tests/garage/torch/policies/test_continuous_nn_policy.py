import numpy as np
import pytest
import torch
from torch import nn

from garage.tf.envs import TfEnv
from garage.torch.modules import MLPModule
from garage.torch.policies import ContinuousNNPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousNNPolicies:
    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (1, 1, (1, )),
        (2, 2, (2, )),
        (3, 3, (3, )),
        (4, 4, (1, 1)),
        (5, 5, (2, 2)),
    ])
    # yapf: enable
    def test_output_mlp_get_action(self, input_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones([1, input_dim], dtype=torch.float32)
        nn_module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        continuous_nn_policy = ContinuousNNPolicy(env_spec, nn_module)
        expected_output = np.full([1, output_dim],
                                  fill_value=input_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(
            continuous_nn_policy.get_action(obs), expected_output)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, input_dim, output_dim, hidden_sizes', [
        (3, 6, 1, (1, )),
        (4, 7, 1, (2, )),
        (5, 8, 2, (3, )),
        (6, 9, 2, (1, 1)),
        (7, 10, 3, (2, 2)),
    ])
    # yapf: enable
    def test_output_mlp_get_actions(self, obs_dim, input_dim, output_dim,
                                    hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones([obs_dim, input_dim], dtype=torch.float32)
        nn_module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        continuous_nn_policy = ContinuousNNPolicy(env_spec, nn_module)
        expected_output = np.full([obs_dim, output_dim],
                                  fill_value=input_dim * np.prod(hidden_sizes),
                                  dtype=np.float32)
        assert np.array_equal(
            continuous_nn_policy.get_actions(obs), expected_output)

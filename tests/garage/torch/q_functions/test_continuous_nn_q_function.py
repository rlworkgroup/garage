import numpy as np
import pytest
import torch
from torch import nn

from garage.tf.envs import TfEnv
from garage.torch.modules import MLPModule
from garage.torch.q_functions import ContinuousNNQFunction
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousNNQFunction:
    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (6, 1, (1, )),
        (7, 1, (2, )),
        (8, 2, (3, )),
        (9, 2, (1, 1)),
        (10, 3, (2, 2)),
    ])
    # yapf: enable
    def test_output_mlp_1d(self, input_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones([1, input_dim], dtype=torch.float32)
        nn_module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        q_function = ContinuousNNQFunction(env_spec, nn_module)
        output = q_function.get_qval(obs)
        expected_output = torch.full(
            [1, output_dim],
            fill_value=input_dim * np.prod(hidden_sizes),
            dtype=torch.float32)

        assert torch.all(torch.eq(output, expected_output))

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, input_dim, output_dim, hidden_sizes', [
        (3, 7, 1, (1, )),
        (4, 8, 1, (2, )),
        (5, 9, 2, (3, )),
        (6, 10, 2, (1, 1)),
        (7, 11, 3, (2, 2)),
    ])
    # yapf: enable
    def test_output_mlp_2d(self, obs_dim, input_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones([obs_dim, input_dim], dtype=torch.float32)
        nn_module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)

        q_function = ContinuousNNQFunction(env_spec, nn_module)
        output = q_function.get_qval(obs)
        expected_output = torch.full(
            [obs_dim, output_dim],
            fill_value=input_dim * np.prod(hidden_sizes),
            dtype=torch.float32)
        assert torch.all(torch.eq(output, expected_output))

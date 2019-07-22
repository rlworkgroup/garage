import pickle

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
    @pytest.mark.parametrize('obs_dim, act_dim, output_dim, hidden_sizes', [
        (1, 1, 1, (1, )),
        (3, 1, 1, (2, )),
        (5, 2, 1, (3, )),
        (6, 2, 1, (1, 1)),
        (7, 3, 1, (2, 2)),
    ])
    # yapf: enable
    def test_get_qval(self, obs_dim, act_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        act = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0)
        nn_module = MLPModule(input_dim=obs_dim + act_dim,
                              output_dim=output_dim,
                              hidden_nonlinearity=None,
                              hidden_sizes=hidden_sizes,
                              hidden_w_init=nn.init.ones_,
                              output_w_init=nn.init.ones_)

        qf = ContinuousNNQFunction(env_spec, nn_module)
        output = qf.get_qval(obs, act)
        expected_output = torch.full([1, output_dim],
                                     fill_value=(obs_dim + act_dim) *
                                     np.prod(hidden_sizes),
                                     dtype=torch.float32)
        assert torch.eq(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, act_dim, output_dim, hidden_sizes', [
        (1, 1, 1, (1, )),
        (3, 1, 1, (2, )),
        (5, 2, 1, (3, )),
        (6, 2, 1, (1, 1)),
        (7, 3, 1, (2, 2)),
    ])
    # yapf: enable
    def test_output_shape(self, obs_dim, act_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        act = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0)
        nn_module = MLPModule(input_dim=obs_dim + act_dim,
                              output_dim=output_dim,
                              hidden_nonlinearity=None,
                              hidden_sizes=hidden_sizes,
                              hidden_w_init=nn.init.ones_,
                              output_w_init=nn.init.ones_)

        qf = ContinuousNNQFunction(env_spec, nn_module)
        output = qf.get_qval(obs, act)

        assert output.shape == (1, 1)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, act_dim, output_dim, hidden_sizes', [
        (1, 1, 1, (1, )),
        (3, 1, 1, (2, )),
        (5, 2, 1, (3, )),
        (6, 2, 1, (1, 1)),
        (7, 3, 1, (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, obs_dim, act_dim, output_dim, hidden_sizes):
        env_spec = TfEnv(DummyBoxEnv())
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        act = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0)
        nn_module = MLPModule(input_dim=obs_dim + act_dim,
                              output_dim=output_dim,
                              hidden_nonlinearity=None,
                              hidden_sizes=hidden_sizes,
                              hidden_w_init=nn.init.ones_,
                              output_w_init=nn.init.ones_)

        qf = ContinuousNNQFunction(env_spec, nn_module)
        output1 = qf.get_qval(obs, act)

        p = pickle.dumps(qf)
        qf_pickled = pickle.loads(p)
        output2 = qf_pickled.get_qval(obs, act)

        assert torch.eq(output1, output2)

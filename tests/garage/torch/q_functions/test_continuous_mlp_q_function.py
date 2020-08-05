import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GymEnv
from garage.torch.q_functions import ContinuousMLPQFunction

from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousNNQFunction:
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 1), (2, 2)])
    # yapf: enable
    def test_forward(self, hidden_sizes):
        env_spec = GymEnv(DummyBoxEnv()).spec
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        act = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0)

        qf = ContinuousMLPQFunction(env_spec=env_spec,
                                    hidden_nonlinearity=None,
                                    hidden_sizes=hidden_sizes,
                                    hidden_w_init=nn.init.ones_,
                                    output_w_init=nn.init.ones_)

        output = qf(obs, act)
        expected_output = torch.full([1, 1],
                                     fill_value=(obs_dim + act_dim) *
                                     np.prod(hidden_sizes),
                                     dtype=torch.float32)
        assert torch.eq(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (3, (2, )),
        (9, (3, )),
        (15, (1, 1)),
        (22, (2, 2)),
    ])
    # yapf: enable
    def test_output_shape(self, batch_size, hidden_sizes):
        env_spec = GymEnv(DummyBoxEnv()).spec
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones(batch_size, obs_dim, dtype=torch.float32)
        act = torch.ones(batch_size, act_dim, dtype=torch.float32)

        qf = ContinuousMLPQFunction(env_spec=env_spec,
                                    hidden_nonlinearity=None,
                                    hidden_sizes=hidden_sizes,
                                    hidden_w_init=nn.init.ones_,
                                    output_w_init=nn.init.ones_)
        output = qf(obs, act)

        assert output.shape == (batch_size, 1)

    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 5), (2, 7, 10)])
    # yapf: enable
    def test_is_pickleable(self, hidden_sizes):
        env_spec = GymEnv(DummyBoxEnv()).spec
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        act = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0)

        qf = ContinuousMLPQFunction(env_spec=env_spec,
                                    hidden_nonlinearity=None,
                                    hidden_sizes=hidden_sizes,
                                    hidden_w_init=nn.init.ones_,
                                    output_w_init=nn.init.ones_)

        output1 = qf(obs, act)

        p = pickle.dumps(qf)
        qf_pickled = pickle.loads(p)
        output2 = qf_pickled(obs, act)

        assert torch.eq(output1, output2)

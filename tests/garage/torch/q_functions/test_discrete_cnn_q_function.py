import pickle

import pytest
import torch
from torch import nn

from garage.envs import GymEnv
from garage.torch.q_functions import DiscreteCNNQFunction

from tests.fixtures.envs.dummy import DummyBoxEnv


# yapf: disable
@pytest.mark.parametrize('batch_size, hidden_channels, kernel_sizes, '
                         'strides',
                         [(1, (32, ), (1, ), (1, )),
                          (3, (32, ), (3, ), (1, )),
                          (9, (32, ), (3, ), (1, )),
                          (15, (32, 64), (1, 1), (1, 1)),
                          (22, (32, 64), (3, 3), (1, 1))])
# yapf: enable
def test_forward(batch_size, hidden_channels, kernel_sizes, strides):
    env_spec = GymEnv(DummyBoxEnv(obs_dim=(3, 10, 10))).spec
    obs_dim = env_spec.observation_space.shape
    obs = torch.zeros((batch_size, ) + obs_dim, dtype=torch.float32)

    qf = DiscreteCNNQFunction(env_spec=env_spec,
                              kernel_sizes=kernel_sizes,
                              strides=strides,
                              mlp_hidden_nonlinearity=None,
                              cnn_hidden_nonlinearity=None,
                              hidden_channels=hidden_channels,
                              hidden_sizes=hidden_channels,
                              hidden_w_init=nn.init.ones_,
                              output_w_init=nn.init.ones_,
                              is_image=False)

    output = qf(obs)
    expected_output = torch.zeros(output.shape)

    assert output.shape == (batch_size, env_spec.action_space.flat_dim)
    assert torch.eq(output, expected_output).all()


# yapf: disable
@pytest.mark.parametrize('batch_size, hidden_channels, '
                         'kernel_sizes, strides',
                         [(1, (32, ), (1, ), (1, )),
                          (15, (32, 64), (1, 1), (1, 1))])
# yapf: enable
def test_is_pickleable(batch_size, hidden_channels, kernel_sizes, strides):
    env_spec = GymEnv(DummyBoxEnv(obs_dim=(3, 10, 10))).spec
    obs_dim = env_spec.observation_space.shape
    obs = torch.ones((batch_size, ) + obs_dim, dtype=torch.float32)

    qf = DiscreteCNNQFunction(env_spec=env_spec,
                              kernel_sizes=kernel_sizes,
                              strides=strides,
                              mlp_hidden_nonlinearity=None,
                              cnn_hidden_nonlinearity=None,
                              hidden_channels=hidden_channels,
                              hidden_sizes=hidden_channels,
                              hidden_w_init=nn.init.ones_,
                              output_w_init=nn.init.ones_,
                              is_image=False)

    output1 = qf(obs)
    p = pickle.dumps(qf)
    qf_pickled = pickle.loads(p)
    output2 = qf_pickled(obs)

    assert torch.eq(output1, output2).all()

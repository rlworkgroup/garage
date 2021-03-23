"""Test DiscreteCNNModule."""
import pickle

import akro
import numpy as np
import pytest
import torch
import torch.nn as nn

from garage import InOutSpec
from garage.torch.modules import CNNModule, DiscreteCNNModule, MLPModule


@pytest.mark.parametrize(
    'output_dim, kernel_sizes, hidden_channels, strides, paddings', [
        (1, (1, ), (32, ), (1, ), (0, )),
        (2, (3, ), (32, ), (1, ), (0, )),
        (5, (3, ), (32, ), (2, ), (0, )),
        (5, (5, ), (12, ), (1, ), (2, )),
        (5, (1, 1), (32, 64), (1, 1), (0, 0)),
        (10, (3, 3), (32, 64), (1, 1), (0, 0)),
        (10, (3, 3), (32, 64), (2, 2), (0, 0)),
    ])
def test_output_values(output_dim, kernel_sizes, hidden_channels, strides,
                       paddings):

    input_width = 32
    input_height = 32
    in_channel = 3
    input_shape = (in_channel, input_height, input_width)
    spec = InOutSpec(akro.Box(shape=input_shape, low=-np.inf, high=np.inf),
                     akro.Box(shape=(output_dim, ), low=-np.inf, high=np.inf))
    obs = torch.rand(input_shape)

    module = DiscreteCNNModule(spec=spec,
                               image_format='NCHW',
                               hidden_channels=hidden_channels,
                               hidden_sizes=hidden_channels,
                               kernel_sizes=kernel_sizes,
                               strides=strides,
                               paddings=paddings,
                               padding_mode='zeros',
                               hidden_w_init=nn.init.ones_,
                               output_w_init=nn.init.ones_)

    cnn = CNNModule(spec=InOutSpec(
        akro.Box(shape=input_shape, low=-np.inf, high=np.inf), None),
                    image_format='NCHW',
                    hidden_channels=hidden_channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
                    padding_mode='zeros',
                    hidden_w_init=nn.init.ones_)
    flat_dim = torch.flatten(cnn(obs).detach(), start_dim=1).shape[1]

    mlp = MLPModule(
        flat_dim,
        output_dim,
        hidden_channels,
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_,
    )

    cnn_out = cnn(obs)
    output = mlp(torch.flatten(cnn_out, start_dim=1))

    assert torch.all(torch.eq(output.detach(), module(obs).detach()))


@pytest.mark.parametrize('output_dim, hidden_channels, kernel_sizes, strides',
                         [(1, (32, ), (1, ), (1, ))])
def test_without_nonlinearity(output_dim, hidden_channels, kernel_sizes,
                              strides):
    input_width = 32
    input_height = 32
    in_channel = 3
    input_shape = (in_channel, input_height, input_width)
    spec = InOutSpec(akro.Box(shape=input_shape, low=-np.inf, high=np.inf),
                     akro.Box(shape=(output_dim, ), low=-np.inf, high=np.inf))

    module = DiscreteCNNModule(spec=spec,
                               image_format='NCHW',
                               hidden_channels=hidden_channels,
                               hidden_sizes=hidden_channels,
                               kernel_sizes=kernel_sizes,
                               strides=strides,
                               mlp_hidden_nonlinearity=None,
                               cnn_hidden_nonlinearity=None,
                               hidden_w_init=nn.init.ones_,
                               output_w_init=nn.init.ones_)

    assert len(module._module) == 3


@pytest.mark.parametrize('output_dim, hidden_channels, kernel_sizes, strides',
                         [(1, (32, ), (1, ), (1, )), (5, (32, ), (3, ), (1, )),
                          (2, (32, ), (3, ), (1, )),
                          (3, (32, 64), (1, 1), (1, 1)),
                          (4, (32, 64), (3, 3), (1, 1))])
def test_is_pickleable(output_dim, hidden_channels, kernel_sizes, strides):
    input_width = 32
    input_height = 32
    in_channel = 3
    input_shape = (in_channel, input_height, input_width)
    input_a = torch.ones(input_shape)
    spec = InOutSpec(akro.Box(shape=input_shape, low=-np.inf, high=np.inf),
                     akro.Box(shape=(output_dim, ), low=-np.inf, high=np.inf))

    model = DiscreteCNNModule(spec=spec,
                              image_format='NCHW',
                              hidden_channels=hidden_channels,
                              kernel_sizes=kernel_sizes,
                              mlp_hidden_nonlinearity=nn.ReLU,
                              cnn_hidden_nonlinearity=nn.ReLU,
                              strides=strides)
    output1 = model(input_a)

    h = pickle.dumps(model)
    model_pickled = pickle.loads(h)
    output2 = model_pickled(input_a)

    assert np.array_equal(torch.all(torch.eq(output1, output2)), True)

"""Test NoisyMLPModule."""
import math

import numpy as np
import pytest
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from garage.torch.modules import NoisyMLPModule
from garage.torch.modules.noisy_mlp_module import NoisyLinear


@pytest.mark.parametrize('input_dim, output_dim, sigma_naught, hidden_sizes',
                         [(1, 1, 0.1, (32, 32)), (2, 2, 0.5, (32, 64)),
                          (2, 3, 1., (5, 5, 5))])
def test_forward(input_dim, output_dim, sigma_naught, hidden_sizes):
    noisy_mlp = NoisyMLPModule(input_dim,
                               output_dim,
                               hidden_nonlinearity=None,
                               sigma_naught=sigma_naught,
                               hidden_sizes=hidden_sizes)

    # mock the noise
    for layer in noisy_mlp._noisy_layers:
        layer._get_noise = lambda x: torch.Tensor(np.ones(x))

    input_val = torch.Tensor(np.ones(input_dim, dtype=np.float32))
    x = input_val
    for layer in noisy_mlp._noisy_layers:
        x = layer(x)

    out = noisy_mlp.forward(input_val)
    assert (x == out).all()


@pytest.mark.parametrize('input_dim, output_dim, sigma_naught, hidden_sizes',
                         [(1, 1, 0.1, (32, 32)), (2, 2, 0.5, (32, 64)),
                          (2, 3, 1., (5, 5, 5))])
def test_forward_deterministic(input_dim, output_dim, sigma_naught,
                               hidden_sizes):
    noisy_mlp = NoisyMLPModule(input_dim,
                               output_dim,
                               hidden_nonlinearity=None,
                               sigma_naught=sigma_naught,
                               hidden_sizes=hidden_sizes)
    noisy_mlp.set_deterministic(True)
    input_val = torch.Tensor(np.ones(input_dim, dtype=np.float32))
    x = input_val
    for layer in noisy_mlp._noisy_layers:
        x = layer(x)

    out = noisy_mlp.forward(input_val)
    assert (x == out).all()


@pytest.mark.parametrize('input_dim, output_dim, sigma_naught', [(1, 1, 0.1),
                                                                 (2, 2, 0.5),
                                                                 (2, 3, 1.)])
def test_noisy_linear_reset_parameters(input_dim, output_dim, sigma_naught):
    layer = NoisyLinear(input_dim, output_dim, sigma_naught=0)

    mu_range = 1 / math.sqrt(input_dim)
    assert (layer._weight_sigma == 0.).all()
    assert (layer._bias_sigma == 0.).all()

    layer._sigma_naught = sigma_naught
    layer.reset_parameters()

    bias_sig = sigma_naught / math.sqrt(output_dim)
    weight_sig = sigma_naught / math.sqrt(input_dim)

    # sigma
    assert (layer._weight_sigma == weight_sig).all()
    assert (layer._bias_sigma == bias_sig).all()

    # mu
    assert ((layer._bias_mu <= mu_range).all()
            and (layer._bias_mu >= -mu_range).all())

    assert ((layer._weight_mu <= mu_range).all()
            and (layer._weight_mu >= -mu_range).all())


@pytest.mark.parametrize('input_dim, output_dim, sigma_naught', [(1, 1, 0.1),
                                                                 (2, 2, 0.5),
                                                                 (2, 3, 1.)])
def test_noisy_linear_forward(input_dim, output_dim, sigma_naught):
    layer = NoisyLinear(input_dim, output_dim, sigma_naught=sigma_naught)

    input_val = torch.Tensor(np.ones(input_dim, dtype=np.float32))
    val = layer.forward(input_val).detach()
    w = layer._weight_mu + \
        layer._weight_sigma.mul(Variable(layer.weight_epsilon))
    b = layer._bias_mu + layer._bias_sigma.mul(Variable(layer.bias_epsilon))

    expected = F.linear(input_val, w, b).detach()

    assert (val == expected).all()

    # test deterministic mode

    layer.set_deterministic(True)
    val = layer.forward(input_val)
    expected = F.linear(input_val, layer._weight_mu, layer._bias_mu)

    assert (val == expected).all()


@pytest.mark.parametrize('input_dim, output_dim', [(1, 1), (2, 2), (2, 3)])
def test_sample_noise(input_dim, output_dim):
    layer = NoisyLinear(input_dim, output_dim)

    # mock the noise
    layer._get_noise = lambda x: torch.Tensor(np.ones(x))

    out_noise = layer._get_noise(output_dim)
    in_noise = layer._get_noise(input_dim)
    layer._sample_noise()

    expected = out_noise.ger(in_noise).detach()
    assert (layer.weight_epsilon == expected).all()
    assert (layer.bias_epsilon == layer._get_noise(output_dim)).all()

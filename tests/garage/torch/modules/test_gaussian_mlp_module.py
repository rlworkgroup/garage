import pytest
import torch
from torch import nn

from garage.torch.modules.gaussian_mlp_module \
    import GaussianMLPIndependentStdModule, GaussianMLPModule, \
    GaussianMLPTwoHeadedModule

plain_settings = [
    (1, 1, (1, )),
    (1, 2, (2, )),
    (1, 3, (3, )),
    (1, 1, (1, 2)),
    (1, 2, (2, 1)),
    (1, 3, (4, 5)),
    (2, 1, (1, )),
    (2, 2, (2, )),
    (2, 3, (3, )),
    (2, 1, (1, 2)),
    (2, 2, (2, 1)),
    (2, 3, (4, 5)),
    (5, 1, (1, )),
    (5, 2, (2, )),
    (5, 3, (3, )),
    (5, 1, (1, 2)),
    (5, 2, (2, 1)),
    (5, 3, (4, 5)),
]

different_std_settings = [(1, 1, (1, ), (1, )), (1, 2, (2, ), (2, )),
                          (1, 3, (3, ), (3, )), (1, 1, (1, 2), (1, 2)),
                          (1, 2, (2, 1), (2, 1)), (1, 3, (4, 5), (4, 5)),
                          (2, 1, (1, ), (1, )), (2, 2, (2, ), (2, )),
                          (2, 3, (3, ), (3, )), (2, 1, (1, 2), (1, 2)),
                          (2, 2, (2, 1), (2, 1)), (2, 3, (4, 5), (4, 5)),
                          (5, 1, (1, ), (1, )), (5, 2, (2, ), (2, )),
                          (5, 3, (3, ), (3, )), (5, 1, (1, 2), (1, 2)),
                          (5, 2, (2, 1), (2, 1)), (5, 3, (4, 5), (4, 5))]


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_std_share_network_output_values(input_dim, output_dim, hidden_sizes):
    module = GaussianMLPTwoHeadedModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    dist = module(torch.ones(input_dim))

    exp_mean = torch.full(
        (output_dim, ), input_dim * (torch.Tensor(hidden_sizes).prod().item()))
    exp_variance = (
        input_dim * torch.Tensor(hidden_sizes).prod()).exp().pow(2).item()

    assert dist.mean.equal(exp_mean)
    assert dist.variance.equal(torch.full((output_dim, ), exp_variance))
    assert dist.rsample().shape == (output_dim, )


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_std_share_network_output_values_with_batch(input_dim, output_dim,
                                                    hidden_sizes):
    module = GaussianMLPTwoHeadedModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    batch_size = 5
    dist = module(torch.ones([batch_size, input_dim]))

    exp_mean = torch.full(
        (batch_size, output_dim),
        input_dim * (torch.Tensor(hidden_sizes).prod().item()))
    exp_variance = (
        input_dim * torch.Tensor(hidden_sizes).prod()).exp().pow(2).item()

    assert dist.mean.equal(exp_mean)
    assert dist.variance.equal(
        torch.full((batch_size, output_dim), exp_variance))
    assert dist.rsample().shape == (batch_size, output_dim)


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_std_network_output_values(input_dim, output_dim, hidden_sizes):
    init_std = 2.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=init_std,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    dist = module(torch.ones(input_dim))

    exp_mean = torch.full(
        (output_dim, ), input_dim * (torch.Tensor(hidden_sizes).prod().item()))
    exp_variance = init_std**2

    assert dist.mean.equal(exp_mean)
    assert dist.variance.equal(torch.full((output_dim, ), exp_variance))
    assert dist.rsample().shape == (output_dim, )


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_std_network_output_values_with_batch(input_dim, output_dim,
                                              hidden_sizes):
    init_std = 2.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=init_std,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    batch_size = 5
    dist = module(torch.ones([batch_size, input_dim]))

    exp_mean = torch.full(
        (batch_size, output_dim),
        input_dim * (torch.Tensor(hidden_sizes).prod().item()))
    exp_variance = init_std**2

    assert dist.mean.equal(exp_mean)
    assert dist.variance.equal(
        torch.full((batch_size, output_dim), exp_variance))
    assert dist.rsample().shape == (batch_size, output_dim)


@pytest.mark.parametrize(
    'input_dim, output_dim, hidden_sizes, std_hidden_sizes',
    different_std_settings)
def test_std_adaptive_network_output_values(input_dim, output_dim,
                                            hidden_sizes, std_hidden_sizes):
    module = GaussianMLPIndependentStdModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        std_hidden_sizes=std_hidden_sizes,
        hidden_nonlinearity=None,
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_,
        std_hidden_nonlinearity=None,
        std_hidden_w_init=nn.init.ones_,
        std_output_w_init=nn.init.ones_)

    dist = module(torch.ones(input_dim))

    exp_mean = torch.full(
        (output_dim, ), input_dim * (torch.Tensor(hidden_sizes).prod().item()))
    exp_variance = (
        input_dim * torch.Tensor(hidden_sizes).prod()).exp().pow(2).item()

    assert dist.mean.equal(exp_mean)
    assert dist.variance.equal(torch.full((output_dim, ), exp_variance))
    assert dist.rsample().shape == (output_dim, )


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_softplus_std_network_output_values(input_dim, output_dim,
                                            hidden_sizes):
    init_std = 2.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=init_std,
        hidden_nonlinearity=None,
        std_parameterization='softplus',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    dist = module(torch.ones(input_dim))

    exp_mean = input_dim * torch.Tensor(hidden_sizes).prod().item()
    exp_variance = torch.Tensor([init_std]).exp().add(1.).log()**2

    assert dist.mean.equal(torch.full((output_dim, ), exp_mean))
    assert dist.variance.equal(torch.full((output_dim, ), exp_variance[0]))
    assert dist.rsample().shape == (output_dim, )


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_exp_min_std(input_dim, output_dim, hidden_sizes):
    min_value = 10.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=1.,
        min_std=min_value,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.zeros_,
        output_w_init=nn.init.zeros_)

    dist = module(torch.ones(input_dim))

    exp_variance = min_value**2

    assert dist.variance.equal(torch.full((output_dim, ), exp_variance))


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_exp_max_std(input_dim, output_dim, hidden_sizes):
    max_value = 1.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=10.,
        max_std=max_value,
        hidden_nonlinearity=None,
        std_parameterization='exp',
        hidden_w_init=nn.init.zeros_,
        output_w_init=nn.init.zeros_)

    dist = module(torch.ones(input_dim))

    exp_variance = max_value**2

    assert dist.variance.equal(torch.full((output_dim, ), exp_variance))


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_softplus_min_std(input_dim, output_dim, hidden_sizes):
    min_value = 2.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=1.,
        min_std=min_value,
        hidden_nonlinearity=None,
        std_parameterization='softplus',
        hidden_w_init=nn.init.zeros_,
        output_w_init=nn.init.zeros_)

    dist = module(torch.ones(input_dim))

    exp_variance = torch.Tensor([min_value]).exp().add(1.).log()**2

    assert dist.variance.equal(torch.full((output_dim, ), exp_variance[0]))


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', plain_settings)
def test_softplus_max_std(input_dim, output_dim, hidden_sizes):
    max_value = 1.

    module = GaussianMLPModule(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        init_std=10,
        max_std=max_value,
        hidden_nonlinearity=None,
        std_parameterization='softplus',
        hidden_w_init=nn.init.ones_,
        output_w_init=nn.init.ones_)

    dist = module(torch.ones(input_dim))

    exp_variance = torch.Tensor([max_value]).exp().add(1.).log()**2

    assert torch.equal(dist.variance,
                       torch.full((output_dim, ), exp_variance[0]))


def test_unknown_std_parameterization():
    with pytest.raises(NotImplementedError):
        GaussianMLPModule(
            input_dim=1, output_dim=1, std_parameterization='unknown')

import pytest
import torch
from torch import nn

from garage.torch.modules import MultiHeadedMLPModule

plain_settings = [
    (1, 1, (1, ), (0, 1, 1, 2, 3, 5, 5)),
    (1, 2, (2, ), (0, 1, 2)),
    (1, 3, (3, ), (0, 1, 2, 3, 5, 5)),
    (1, 1, (1, 2), (0, 1, 1, 2, 3, 5, 5)),
    (1, 2, (2, 1), (0, 1, 1, 2, 5, 5)),
    (1, 3, (4, 5), (0, 1, 2, 3, 5, 5)),
    (2, 1, (1, ), (0, 1, 1, 3, 5, 5)),
    (2, 2, (2, ), (0, 1, 1, 2, 3, 5, 5)),
    (2, 3, (3, ), (1, 1, 2, 5, 5)),
    (2, 1, (1, 2), (0, 1, 2, 3, 5, 5)),
    (2, 2, (2, 1), (0, 1, 1, 2, 3, 5, 5)),
    (2, 3, (4, 5), (0, 1, 1, 2, 3, 5)),
    (5, 1, (1, ), (0, 1, 1, 2, 3, 5)),
    (5, 2, (2, ), (0, 1, 3, 5, 5)),
    (5, 3, (3, ), (0, 1, 1, 2, 3)),
    (5, 1, (1, 2), (0, 1, 1, 2, 5, 5)),
    (5, 2, (2, 1), (0, 1, 1, 2, 3, 5, 5)),
    (5, 3, (4, 5), (0, 1, 2, 3, 5, 5)),
]

different_output_dims_settings = [
    (1, (1, 4), (1, ), (0, 1)),
    (1, (2, 3), (2, ), (0, 3)),
    (1, (3, 2), (3, ), (5, )),
    (1, (1, 6), (1, 2), (6, 4)),
    (1, (2, 7), (2, 1), (5, 5)),
    (1, (3, 4), (4, 5), (1, 2)),
    (2, (1, 5, 6), (1, ), (6, 1, 3)),
    (2, (2, 1, 3), (2, ), (1, 2, 3)),
    (2, (3, 6, 7), (3, ), (6, )),
    (2, (1, ), (1, 2), (9, )),
    (2, (2, ), (2, 1), (4, )),
    (2, (3, ), (4, 5), (5, )),
    (5, (1, 3, 1), (1, ), (5, )),
    (5, (2, 6), (2, ), (4, 1)),
    (5, (3, 5), (3, ), (2, 5)),
    (5, (1, 3), (1, 2), (6, 8)),
    (5, (2, 1, 1, 1), (2, 1), (1, )),
    (5, (3, 4, 1, 2), (4, 5), (5, 1, 2, 3)),
]

invalid_settings = [
    (1, (1, 4, 5), (1, ), 2, (None, ), (1, 2),
     (nn.init.ones_, )),  # n_head != output_dims
    (1, (1, 4), (1, ), 2, (None, ), (1, 2, 3),
     (nn.init.ones_, )),  # n_head != w_init
    (1, (1, 4), (1, ), 2, (None, None, None), (1, 2),
     (nn.init.ones_, )),  # n_head != nonlinearity
    (1, (1, 4), (1, ), 2, (None, ), (1, 2),
     (nn.init.ones_, nn.init.ones_, nn.init.ones_)),  # n_head != b_init
    (1, (1, 4, 5), (1, ), 3, (None, ), (1, 2),
     (nn.init.ones_, )),  # output_dims > w_init
    (1, (1, ), (1, ), 1, (None, ), (1, 2, 3),
     (nn.init.ones_, )),  # output_dims < w_init
    (1, (1, 4, 5), (1, ), 3, (None, None), (1, 2, 3),
     (nn.init.ones_, )),  # output_dims > nonlinearity
    (1, (1, ), (1, ), 1, (None, None, None), (1, 2, 3),
     (nn.init.ones_, )),  # output_dims < nonlinearity
    (1, (1, 4, 5), (1, ), 3, (None, ), (1, 2, 3),
     (nn.init.ones_, nn.init.ones_)),  # output_dims > b_init
    (1, (1, ), (1, ), 1, (None, ), (1, 2, 3),
     (nn.init.ones_, nn.init.ones_, nn.init.ones_)),  # output_dims > b_init
]


def helper_make_inits(val):
    return lambda x: nn.init.constant_(x, val)


@pytest.mark.parametrize(
    'input_dim, output_dim, hidden_sizes, output_w_init_vals', plain_settings)
def test_multi_headed_mlp_module(input_dim, output_dim, hidden_sizes,
                                 output_w_init_vals):
    module = MultiHeadedMLPModule(
        n_heads=len(output_w_init_vals),
        input_dim=input_dim,
        output_dims=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=None,
        hidden_w_init=nn.init.ones_,
        output_nonlinearities=None,
        output_w_inits=list(map(helper_make_inits, output_w_init_vals)))

    input = torch.ones(input_dim)
    outputs = module(input)

    for i in range(len(outputs)):
        expected = input_dim * torch.Tensor(hidden_sizes).prod()
        expected *= output_w_init_vals[i]
        assert torch.equal(outputs[i], torch.full((output_dim, ), expected))


@pytest.mark.parametrize(
    'input_dim, output_dim, hidden_sizes, output_w_init_vals',
    different_output_dims_settings)
def test_multi_headed_mlp_module_with_different_output_dims(
        input_dim, output_dim, hidden_sizes, output_w_init_vals):
    module = MultiHeadedMLPModule(
        n_heads=len(output_dim),
        input_dim=input_dim,
        output_dims=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=None,
        hidden_w_init=nn.init.ones_,
        output_nonlinearities=None,
        output_w_inits=list(map(helper_make_inits, output_w_init_vals)))

    input = torch.ones(input_dim)
    outputs = module(input)

    if len(output_w_init_vals) == 1:
        output_w_init_vals = output_w_init_vals * len(output_dim)
    for i in range(len(outputs)):
        expected = input_dim * torch.Tensor(hidden_sizes).prod()
        expected *= output_w_init_vals[i]
        assert torch.equal(outputs[i], torch.full((output_dim[i], ), expected))


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes, '
                         'n_heads, nonlinearity, w_init, b_init',
                         invalid_settings)
def test_invalid_settings(input_dim, output_dim, hidden_sizes, n_heads,
                          nonlinearity, w_init, b_init):
    with pytest.raises(ValueError):
        MultiHeadedMLPModule(
            n_heads=n_heads,
            input_dim=input_dim,
            output_dims=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=None,
            hidden_w_init=nn.init.ones_,
            output_nonlinearities=nonlinearity,
            output_w_inits=list(map(helper_make_inits, w_init)),
            output_b_inits=b_init)

"""Test Multi-headed MLPModule."""
import pytest
import torch
from torch import nn

from garage.torch.modules import MultiHeadedMLPModule

plain_settings = [
    (1, (2, ), (2, ), (0, 1, 2), 3),
    (5, (3, ), (4, 5), (0, 1, 2, 3, 5, 5), 6),
    (1, (2, 3), (2, ), (0, 3), 2),
    (2, (3, 6, 7), (3, ), (6, ), 3),
    (5, (3, 4, 1, 2), (4, 5), (5, 1, 2, 3), 4),
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


def _helper_make_inits(val):
    """Return the function that initialize variable with val.

    Args:
        val (int): Value to initialize variable.

    Returns:
        lambda: Lambda function that initialize variable with val.

    """
    return lambda x: nn.init.constant_(x, val)


@pytest.mark.parametrize(
    'input_dim, output_dim, hidden_sizes, output_w_init_vals, n_heads',
    plain_settings)
def test_multi_headed_mlp_module(input_dim, output_dim, hidden_sizes,
                                 output_w_init_vals, n_heads):
    """Test Multi-headed MLPModule.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Ouput dimension.
        hidden_sizes (list[int]): Size of hidden layers.
        output_w_init_vals (list[int]): Init values for output weights.
        n_heads (int): Number of output layers.

    """
    module = MultiHeadedMLPModule(n_heads=n_heads,
                                  input_dim=input_dim,
                                  output_dims=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  hidden_nonlinearity=None,
                                  hidden_w_init=nn.init.ones_,
                                  output_nonlinearities=None,
                                  output_w_inits=list(
                                      map(_helper_make_inits,
                                          output_w_init_vals)))

    input_value = torch.ones(input_dim)
    outputs = module(input_value)

    if len(output_w_init_vals) == 1:
        output_w_init_vals = list(output_w_init_vals) * n_heads
    if len(output_dim) == 1:
        output_dim = list(output_dim) * n_heads
    for i, output in enumerate(outputs):
        expected = input_dim * torch.Tensor(hidden_sizes).prod()
        expected *= output_w_init_vals[i]
        assert torch.equal(
            output, torch.full((output_dim[i], ), expected, dtype=torch.float))


@pytest.mark.parametrize(
    'input_dim, output_dim, hidden_sizes, output_w_init_vals, n_heads',
    plain_settings)
def test_multi_headed_mlp_module_with_layernorm(input_dim, output_dim,
                                                hidden_sizes,
                                                output_w_init_vals, n_heads):
    """Test Multi-headed MLPModule with layer normalization.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Ouput dimension.
        hidden_sizes (list[int]): Size of hidden layers.
        output_w_init_vals (list[int]): Init values for output weights.
        n_heads (int): Number of output layers.

    """
    module = MultiHeadedMLPModule(n_heads=n_heads,
                                  input_dim=input_dim,
                                  output_dims=output_dim,
                                  hidden_sizes=hidden_sizes,
                                  hidden_nonlinearity=None,
                                  layer_normalization=True,
                                  hidden_w_init=nn.init.ones_,
                                  output_nonlinearities=None,
                                  output_w_inits=list(
                                      map(_helper_make_inits,
                                          output_w_init_vals)))

    input_value = torch.ones(input_dim)
    outputs = module(input_value)

    if len(output_w_init_vals) == 1:
        output_w_init_vals = list(output_w_init_vals) * n_heads
    if len(output_dim) == 1:
        output_dim = list(output_dim) * n_heads
    for i, output in enumerate(outputs):
        expected = input_dim * torch.Tensor(hidden_sizes).prod()
        expected *= output_w_init_vals[i]
        assert torch.equal(output, torch.zeros(output_dim[i]))


@pytest.mark.parametrize('input_dim, output_dim, hidden_sizes, '
                         'n_heads, nonlinearity, w_init, b_init',
                         invalid_settings)
def test_invalid_settings(input_dim, output_dim, hidden_sizes, n_heads,
                          nonlinearity, w_init, b_init):
    """Test Multi-headed MLPModule with invalid parameters.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Ouput dimension.
        hidden_sizes (list[int]): Size of hidden layers.
        n_heads (int): Number of output layers.
        nonlinearity (callable or torch.nn.Module): Non-linear functions for
            output layers
        w_init (list[callable]): Initializer function for the weight in
            output layer.
        b_init (list[callable]): Initializer function for the bias in
            output layer.

    """
    expected_msg_template = ('should be either an integer or a collection of '
                             'length n_heads')
    with pytest.raises(ValueError, match=expected_msg_template):
        MultiHeadedMLPModule(n_heads=n_heads,
                             input_dim=input_dim,
                             output_dims=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=None,
                             hidden_w_init=nn.init.ones_,
                             output_nonlinearities=nonlinearity,
                             output_w_inits=list(
                                 map(_helper_make_inits, w_init)),
                             output_b_inits=b_init)

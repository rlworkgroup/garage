"""Module to test garage.torch._functions."""

import numpy as np
import torch

from garage.torch import _GPU_ID, _USE_GPU
from garage.torch import dict_np_to_torch, global_device
from garage.torch import product_of_gaussians, set_gpu_mode, torch_to_np


def test_utils_set_gpu_mode():
    """Test setting gpu mode to False to force CPU."""
    if torch.cuda.is_available():
        set_gpu_mode(mode=True)
        assert global_device() == torch.device('cuda:0')
        assert _USE_GPU
    else:
        set_gpu_mode(mode=False)
        assert global_device() == torch.device('cpu')
        assert not _USE_GPU
    assert not _GPU_ID


def test_torch_to_np():
    """Test whether tuples of tensors can be converted to np arrays."""
    tup = (torch.zeros(1), torch.zeros(1))
    np_out_1, np_out_2 = torch_to_np(tup)
    assert isinstance(np_out_1, np.ndarray)
    assert isinstance(np_out_2, np.ndarray)


def test_dict_np_to_torch():
    """Test if dict whose values are tensors can be converted to np arrays."""
    dic = {'a': np.zeros(1), 'b': np.ones(1)}
    dict_np_to_torch(dic)
    for tensor in dic.values():
        assert isinstance(tensor, torch.Tensor)


def test_product_of_gaussians():
    """Test computing mu, sigma of product of gaussians."""
    size = 5
    mu = torch.ones(size)
    sigmas_squared = torch.ones(size)
    output = product_of_gaussians(mu, sigmas_squared)
    assert output[0] == 1
    assert output[1] == 1 / size

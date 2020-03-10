"""Module to test garage.torch.utils."""

import numpy as np
import torch

import garage.torch.utils as tu


def test_utils_set_gpu_mode():
    """Test setting gpu mode to False to force CPU."""
    if torch.cuda.is_available():
        tu.set_gpu_mode(mode=True)
        assert tu.global_device() == torch.device('cuda:0')
        assert tu._USE_GPU
    else:
        tu.set_gpu_mode(mode=False)
        assert tu.global_device() == torch.device('cpu')
        assert not tu._USE_GPU
    assert not tu._GPU_ID


def test_torch_to_np():
    """Test whether tuples of tensors can be converted to np arrays."""
    tup = (torch.zeros(1), torch.zeros(1))
    np_out_1, np_out_2 = tu.torch_to_np(tup)
    assert isinstance(np_out_1, np.ndarray)
    assert isinstance(np_out_2, np.ndarray)


def test_dict_np_to_torch():
    """Test if dict whose values are tensors can be converted to np arrays."""
    dic = {'a': np.zeros(1), 'b': np.ones(1)}
    tu.dict_np_to_torch(dic)
    for tensor in dic.values():
        assert isinstance(tensor, torch.Tensor)


def test_product_of_gaussians():
    """Test computing mu, sigma of product of gaussians."""
    size = 5
    mu = torch.ones(size)
    sigmas_squared = torch.ones(size)
    output = tu.product_of_gaussians(mu, sigmas_squared)
    assert output[0] == 1
    assert output[1] == 1 / size

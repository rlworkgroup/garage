"""Module to test garage.torch._functions."""
# yapf: disable
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from garage.torch import (compute_advantages,
                          dict_np_to_torch,
                          flatten_to_single_vector,
                          global_device,
                          pad_to_last,
                          product_of_gaussians,
                          set_gpu_mode,
                          torch_to_np,
                          TransposeImage)
import garage.torch._functions as tu

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv

# yapf: enable


def stack(d, arr):
    """Stack 'arr' 'd' times."""
    return np.repeat(np.expand_dims(arr, axis=0), repeats=d, axis=0)


ONES = np.ones((6, ))
ZEROS = np.zeros((6, ))
ARRANGE = np.arange(6)
PI_DIGITS = np.array([3, 1, 4, 1, 5, 9])
FIBS = np.array([1, 1, 2, 3, 5, 8])

nums_1d = np.arange(0, 4).astype(float)
nums_2d = np.arange(0, 4).astype(float).reshape(2, 2)
nums_3d = np.arange(0, 8).astype(float).reshape(2, 2, 2)


def test_utils_set_gpu_mode():
    """Test setting gpu mode to False to force CPU."""
    if torch.cuda.is_available():
        set_gpu_mode(mode=True)
        assert global_device() == torch.device('cuda:0')
        assert tu._USE_GPU
    else:
        set_gpu_mode(mode=False)
        assert global_device() == torch.device('cpu')
        assert not tu._USE_GPU
    assert not tu._GPU_ID


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


def test_flatten_to_single_vector():
    """ Test test_flatten_to_single_vector"""
    x = torch.arange(12).view(2, 1, 3, 2)
    flatten_tensor = flatten_to_single_vector(x)
    expected = np.arange(12).reshape(2, 6)
    # expect [[ 0,  1,  2,  3,  4,  5], [ 6,  7,  8,  9, 10, 11]]
    assert torch.Size([2, 6]) == flatten_tensor.size()
    assert expected.shape == flatten_tensor.shape


def test_transpose_image():
    """Test TransposeImage."""
    original_env = DummyDiscretePixelEnv()
    obs_shape = original_env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        transposed_env = TransposeImage(original_env)
        assert (original_env.observation_space.shape[2] ==
                transposed_env.observation_space.shape[0])


class TestTorchAlgoUtils(TfGraphTestCase):
    """Test class for torch algo utility functions."""
    # yapf: disable
    @pytest.mark.parametrize('discount', [1, 0.95])
    @pytest.mark.parametrize('num_eps', [1, 5])
    @pytest.mark.parametrize('gae_lambda', [0, 0.5, 1])
    @pytest.mark.parametrize('rewards_eps, baselines_eps', [
        (ONES, ZEROS),
        (PI_DIGITS, ARRANGE),
        (ONES, FIBS),
    ])
    # yapf: enable
    def test_compute_advantages(self, num_eps, discount, gae_lambda,
                                rewards_eps, baselines_eps):
        """Test compute_advantage function."""

        def get_advantage(discount, gae_lambda, rewards, baselines):
            adv = torch.zeros(rewards.shape)
            for i in range(rewards.shape[0]):
                acc = 0
                for j in range(rewards.shape[1]):
                    acc = acc * discount * gae_lambda
                    acc += rewards[i][-j - 1] - baselines[i][-j - 1]
                    acc += discount * baselines[i][-j] if j else 0
                    adv[i][-j - 1] = acc
            return adv

        length = len(rewards_eps)

        rewards = torch.Tensor(stack(num_eps, rewards_eps))
        baselines = torch.Tensor(stack(num_eps, baselines_eps))
        expected_adv = get_advantage(discount, gae_lambda, rewards, baselines)
        computed_adv = compute_advantages(discount, gae_lambda, length,
                                          baselines, rewards)
        assert torch.allclose(expected_adv, computed_adv)

    def test_add_padding_last_1d(self):
        """Test pad_to_last function for 1d."""
        max_length = 10

        expected = F.pad(torch.Tensor(nums_1d),
                         (0, max_length - nums_1d.shape[-1]))

        tensor_padding = pad_to_last(nums_1d, total_length=max_length)
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_1d, total_length=10, axis=0)
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_2d(self):
        """Test pad_to_last function for 2d."""
        max_length = 10

        tensor_padding = pad_to_last(nums_2d, total_length=10)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_2d, total_length=10, axis=0)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, 0, 0, max_length - nums_2d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_2d, total_length=10, axis=1)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_3d(self):
        """Test pad_to_last function for 3d."""
        max_length = 10

        tensor_padding = pad_to_last(nums_3d, total_length=10)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=0)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, 0, 0, max_length - nums_3d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=1)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, max_length - nums_3d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=2)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

    @pytest.mark.parametrize('nums', [nums_1d, nums_2d, nums_3d])
    def test_out_of_index_error(self, nums):
        """Test pad_to_last raises IndexError."""
        with pytest.raises(IndexError):
            pad_to_last(nums, total_length=10, axis=len(nums.shape))

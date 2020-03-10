"""Test torch algo utility functions."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import garage.torch.algos._utils as torch_algo_utils
from tests.fixtures import TfGraphTestCase


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


class TestTorchAlgoUtils(TfGraphTestCase):
    """Test class for torch algo utility functions."""
    # yapf: disable
    @pytest.mark.parametrize('discount', [1, 0.95])
    @pytest.mark.parametrize('num_trajs', [1, 5])
    @pytest.mark.parametrize('gae_lambda', [0, 0.5, 1])
    @pytest.mark.parametrize('rewards_traj, baselines_traj', [
        (ONES, ZEROS),
        (PI_DIGITS, ARRANGE),
        (ONES, FIBS),
    ])
    # yapf: enable
    def test_compute_advantages(self, num_trajs, discount, gae_lambda,
                                rewards_traj, baselines_traj):
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

        length = len(rewards_traj)

        rewards = torch.Tensor(stack(num_trajs, rewards_traj))
        baselines = torch.Tensor(stack(num_trajs, baselines_traj))
        expected_adv = get_advantage(discount, gae_lambda, rewards, baselines)
        computed_adv = torch_algo_utils.compute_advantages(
            discount, gae_lambda, length, baselines, rewards)
        assert torch.allclose(expected_adv, computed_adv)

    def test_add_padding_last_1d(self):
        """Test pad_to_last function for 1d."""
        max_length = 10

        expected = F.pad(torch.Tensor(nums_1d),
                         (0, max_length - nums_1d.shape[-1]))

        tensor_padding = torch_algo_utils.pad_to_last(nums_1d,
                                                      total_length=max_length)
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_1d,
                                                      total_length=10,
                                                      axis=0)
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_2d(self):
        """Test pad_to_last function for 2d."""
        max_length = 10

        tensor_padding = torch_algo_utils.pad_to_last(nums_2d, total_length=10)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_2d,
                                                      total_length=10,
                                                      axis=0)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, 0, 0, max_length - nums_2d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_2d,
                                                      total_length=10,
                                                      axis=1)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_3d(self):
        """Test pad_to_last function for 3d."""
        max_length = 10

        tensor_padding = torch_algo_utils.pad_to_last(nums_3d, total_length=10)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=0)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, 0, 0, max_length - nums_3d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=1)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, max_length - nums_3d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_algo_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=2)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

    @pytest.mark.parametrize('nums', [nums_1d, nums_2d, nums_3d])
    def test_out_of_index_error(self, nums):
        """Test pad_to_last raises IndexError."""
        with pytest.raises(IndexError):
            torch_algo_utils.pad_to_last(nums,
                                         total_length=10,
                                         axis=len(nums.shape))

    def testmake_optimizer_with_type(self):
        """Test make_optimizer function with type as first argument."""
        optimizer_type = torch.optim.Adam
        module = torch.nn.Linear(2, 1)
        lr = 0.123
        optimizer = torch_algo_utils.make_optimizer(optimizer_type,
                                                    module,
                                                    lr=lr)
        assert isinstance(optimizer, optimizer_type)
        assert optimizer.defaults['lr'] == lr

    def testmake_optimizer_with_tuple(self):
        """Test make_optimizer function with tuple as first argument."""
        optimizer_type = (torch.optim.Adam, {'lr': 0.1})
        module = torch.nn.Linear(2, 1)
        optimizer = torch_algo_utils.make_optimizer(optimizer_type, module)
        assert isinstance(optimizer, optimizer_type)
        assert optimizer.defaults['lr'] == optimizer_type[1]['lr']

    def testmake_optimizer_raise_value_error(self):
        """Test make_optimizer raises value error."""
        optimizer_type = (torch.optim.Adam, {'lr': 0.1})
        module = torch.nn.Linear(2, 1)
        with pytest.raises(ValueError):
            _ = torch_algo_utils.make_optimizer(optimizer_type,
                                                module,
                                                lr=0.123)

import numpy as np
import pytest
import torch

import garage.torch.algos.loss_function_utils as torch_loss_utils
from tests.fixtures import TfGraphTestCase


def stack(d, arr):
    return np.repeat(np.expand_dims(arr, axis=0), repeats=d, axis=0)


ONES = np.ones((6, ))
ZEROS = np.zeros((6, ))
ARRANGE = np.arange(6)
PI_DIGITS = np.array([3, 1, 4, 1, 5, 9])
FIBS = np.array([1, 1, 2, 3, 5, 8])


class TestLossFunctionUtils(TfGraphTestCase):
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
        computed_adv = torch_loss_utils.compute_advantages(
            discount, gae_lambda, length, baselines, rewards)

        assert torch.allclose(expected_adv, computed_adv)

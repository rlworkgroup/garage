import numpy as np
import pytest
import tensorflow as tf
import torch
import torch.nn.functional as F

import garage.tf.misc.tensor_utils as tf_utils
import garage.torch.algos.loss_function_utils as torch_loss_utils
from tests.fixtures import TfGraphTestCase


def stack(d, arr):
    return np.repeat(np.expand_dims(arr, axis=0), repeats=d, axis=0)


ONES = np.ones((4, 6))
ZEROS = np.zeros((4, 6))
ARRANGE = stack(4, np.arange(6))
PI_DIGITS = stack(4, [3, 1, 4, 1, 5, 9])
E_DIGITS = stack(4, [2, 7, 1, 8, 2, 8])
FIBS = stack(4, [1, 1, 2, 3, 5, 8])

nums_1d = np.arange(0, 4).astype(float)
nums_2d = np.arange(0, 4).astype(float).reshape(2, 2)
nums_3d = np.arange(0, 8).astype(float).reshape(2, 2, 2)


class TestLossFunctionUtils(TfGraphTestCase):
    # yapf: disable
    @pytest.mark.parametrize('gae_lambda, rewards_val, baselines_val', [
        (0.4, ONES, ZEROS),
        (0.8, PI_DIGITS, ARRANGE),
        (1.2, ONES, FIBS),
        (1.7, E_DIGITS, PI_DIGITS),
    ])
    # yapf: enable
    def test_compute_advantages(self, gae_lambda, rewards_val, baselines_val):
        discount = 0.99
        max_len = rewards_val.shape[-1]

        torch_advs = torch_loss_utils.compute_advantages(
            discount, gae_lambda, max_len, torch.Tensor(baselines_val),
            torch.Tensor(rewards_val))

        rewards = tf.compat.v1.placeholder(dtype=tf.float32,
                                           name='reward',
                                           shape=[None, None])
        baselines = tf.compat.v1.placeholder(dtype=tf.float32,
                                             name='baseline',
                                             shape=[None, None])
        adv = tf_utils.compute_advantages(discount, gae_lambda, max_len,
                                          baselines, rewards)
        tf_advs = self.sess.run(adv,
                                feed_dict={
                                    rewards: rewards_val,
                                    baselines: baselines_val,
                                })

        assert np.allclose(torch_advs.numpy(),
                           tf_advs.reshape(torch_advs.shape),
                           atol=1e-5)

    def test_add_padding_last_1d(self):
        max_length = 10

        expected = F.pad(torch.Tensor(nums_1d),
                         (0, max_length - nums_1d.shape[-1]))

        tensor_padding = torch_loss_utils.pad_to_last(nums_1d,
                                                      total_length=max_length)
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_1d,
                                                      total_length=10,
                                                      axis=0)
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_2d(self):
        max_length = 10

        tensor_padding = torch_loss_utils.pad_to_last(nums_2d, total_length=10)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_2d,
                                                      total_length=10,
                                                      axis=0)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, 0, 0, max_length - nums_2d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_2d,
                                                      total_length=10,
                                                      axis=1)
        expected = F.pad(torch.Tensor(nums_2d),
                         (0, max_length - nums_2d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_3d(self):
        max_length = 10

        tensor_padding = torch_loss_utils.pad_to_last(nums_3d, total_length=10)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=0)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, 0, 0, max_length - nums_3d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=1)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, 0, 0, max_length - nums_3d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = torch_loss_utils.pad_to_last(nums_3d,
                                                      total_length=10,
                                                      axis=2)
        expected = F.pad(torch.Tensor(nums_3d),
                         (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

    @pytest.mark.parametrize('nums', [nums_1d, nums_2d, nums_3d])
    def test_out_of_index_error(self, nums):
        with pytest.raises(IndexError):
            torch_loss_utils.pad_to_last(nums,
                                         total_length=10,
                                         axis=len(nums.shape))

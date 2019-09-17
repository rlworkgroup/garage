import numpy as np
import pytest
import tensorflow as tf
import torch

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

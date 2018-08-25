"""
This script creates a test that tests functions in garage.tf.misc.tensor_utils.
"""
import unittest

import numpy as np
import tensorflow as tf

from garage.tf.misc.tensor_utils import compute_advantages


class TestTensorUtil(unittest.TestCase):
    def test_compute_advantages(self):
        """Tests compute_advantages function in utils."""
        discount = 1
        gae_lambda = 1
        max_len = 1
        rewards = tf.placeholder(
            dtype=tf.float32, name="reward", shape=[None, None])
        baselines = tf.placeholder(
            dtype=tf.float32, name="baseline", shape=[None, None])
        adv = compute_advantages(discount, gae_lambda, max_len, baselines,
                                 rewards)

        # Set up inputs and outputs
        rewards_val = np.ones(shape=[2, 1])
        baselines_val = np.zeros(shape=[2, 1])
        desired_val = np.array([1., 1.])

        with tf.Session() as sess:
            adv = sess.run(
                adv,
                feed_dict={
                    rewards: rewards_val,
                    baselines: baselines_val,
                })

        assert np.array_equal(adv, desired_val)

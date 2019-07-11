"""
This script creates a test that tests functions in garage.tf.misc.tensor_utils.
"""
import numpy as np
import tensorflow as tf

from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import get_target_ops
from tests.fixtures import TfGraphTestCase


class TestTensorUtil(TfGraphTestCase):
    def test_compute_advantages(self):
        """Tests compute_advantages function in utils."""
        discount = 1
        gae_lambda = 1
        max_len = 1
        rewards = tf.compat.v1.placeholder(
            dtype=tf.float32, name='reward', shape=[None, None])
        baselines = tf.compat.v1.placeholder(
            dtype=tf.float32, name='baseline', shape=[None, None])
        adv = compute_advantages(discount, gae_lambda, max_len, baselines,
                                 rewards)

        # Set up inputs and outputs
        rewards_val = np.ones(shape=[2, 1])
        baselines_val = np.zeros(shape=[2, 1])
        desired_val = np.array([1., 1.])

        adv = self.sess.run(
            adv, feed_dict={
                rewards: rewards_val,
                baselines: baselines_val,
            })

        assert np.array_equal(adv, desired_val)

    def test_get_target_ops(self):
        var = tf.compat.v1.get_variable(
            'var', [1], initializer=tf.constant_initializer(1))
        target_var = tf.compat.v1.get_variable(
            'target_var', [1], initializer=tf.constant_initializer(2))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        assert target_var.eval() == 2
        update_ops = get_target_ops([var], [target_var])
        self.sess.run(update_ops)
        assert target_var.eval() == 1

    def test_get_target_ops_tau(self):
        var = tf.compat.v1.get_variable(
            'var', [1], initializer=tf.constant_initializer(1))
        target_var = tf.compat.v1.get_variable(
            'target_var', [1], initializer=tf.constant_initializer(2))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        assert target_var.eval() == 2
        init_ops, update_ops = get_target_ops([var], [target_var], tau=0.2)
        self.sess.run(update_ops)
        assert np.allclose(target_var.eval(), 1.8)
        self.sess.run(init_ops)
        assert np.allclose(target_var.eval(), 1)

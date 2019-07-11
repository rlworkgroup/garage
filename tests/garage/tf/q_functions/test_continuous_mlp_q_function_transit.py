"""
Unit test for ContinuousMLPQFunctionWithModel.

This test consists of four different ContinuousMLPQFunction: P1, P2, P3
and P4. P1 and P2 are from ContinuousMLPQFunction, which does not use
garage.tf.models.MLPMergeModel while P3 and P4 do use.

This test ensures the outputs from all the Q Functions are the same,
for the transition from using ContinuousMLPQFunction to
ContinuousMLPQFunctionWithModel.

It covers get_q_val and get_qval_sym
"""

from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.q_functions import ContinuousMLPQFunctionWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousMLPQFunctionTransit(TfGraphTestCase):
    def setup_method(self):
        with mock.patch('tensorflow.random.normal') as mock_rand:
            mock_rand.return_value = 0.5
            super().setup_method()
            self.obs_dim = (5, )
            self.act_dim = (2, )
            self.box_env = TfEnv(
                DummyBoxEnv(obs_dim=self.obs_dim, action_dim=self.act_dim))
            self.qf1 = ContinuousMLPQFunction(
                env_spec=self.box_env, hidden_sizes=(32, 32), name='QF1')
            self.qf2 = ContinuousMLPQFunction(
                env_spec=self.box_env, hidden_sizes=(64, 64), name='QF2')
            self.qf3 = ContinuousMLPQFunctionWithModel(
                env_spec=self.box_env, hidden_sizes=(32, 32), name='QF3')
            self.qf4 = ContinuousMLPQFunctionWithModel(
                env_spec=self.box_env, hidden_sizes=(64, 64), name='QF4')

            self.sess.run(tf.compat.v1.global_variables_initializer())

            for a, b in zip(self.qf3.get_trainable_vars(),
                            self.qf1.get_trainable_vars()):
                self.sess.run(a.assign(b))
            for a, b in zip(self.qf4.get_trainable_vars(),
                            self.qf2.get_trainable_vars()):
                self.sess.run(a.assign(b))

            self.obs = self.box_env.reset()
            self.act = np.full((2, ), 0.5)

    def test_get_qval(self):
        q_val1 = self.qf1.get_qval([self.obs], [self.act])
        q_val2 = self.qf2.get_qval([self.obs], [self.act])
        q_val3 = self.qf3.get_qval([self.obs], [self.act])
        q_val4 = self.qf4.get_qval([self.obs], [self.act])

        assert np.array_equal(q_val1, q_val3)
        assert np.array_equal(q_val2, q_val4)

        q_val1 = self.qf1.get_qval([self.obs, self.obs], [self.act, self.act])
        q_val2 = self.qf2.get_qval([self.obs, self.obs], [self.act, self.act])
        q_val3 = self.qf3.get_qval([self.obs, self.obs], [self.act, self.act])
        q_val4 = self.qf4.get_qval([self.obs, self.obs], [self.act, self.act])

        assert np.array_equal(q_val1, q_val3)
        assert np.allclose(q_val2, q_val4)

    def test_get_qval_sym(self):
        obs_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + self.obs_dim)
        act_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + self.act_dim)

        qval_sym1 = self.qf1.get_qval_sym(obs_ph, act_ph, name='qval_sym')
        qval_sym2 = self.qf2.get_qval_sym(obs_ph, act_ph, name='qval_sym')
        qval_sym3 = self.qf3.get_qval_sym(obs_ph, act_ph, name='qval_sym')
        qval_sym4 = self.qf4.get_qval_sym(obs_ph, act_ph, name='qval_sym')

        q_val1 = self.sess.run(
            qval_sym1, feed_dict={
                obs_ph: [self.obs],
                act_ph: [self.act]
            })
        q_val2 = self.sess.run(
            qval_sym2, feed_dict={
                obs_ph: [self.obs],
                act_ph: [self.act]
            })
        q_val3 = self.sess.run(
            qval_sym3, feed_dict={
                obs_ph: [self.obs],
                act_ph: [self.act]
            })
        q_val4 = self.sess.run(
            qval_sym4, feed_dict={
                obs_ph: [self.obs],
                act_ph: [self.act]
            })

        assert np.array_equal(q_val1, q_val3)
        assert np.array_equal(q_val2, q_val4)

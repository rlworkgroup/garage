import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import CategoricalCNNPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestCategoricalCNNPolicyWithModel(TfGraphTestCase):

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (3, 3)), ), (1, ), 'VALID', (4, )),
        (((3, (3, 3)), (3, (3, 3))), (1, 1), 'VALID', (4, 4)),
        (((3, (3, 3)), (3, (3, 3))), (2, 2), 'SAME', (4, 4)),
    ])
    def test_get_action(self, filters, strides, padding, hidden_sizes):
        env = GarageEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (3, 3)), ), (1, ), 'VALID', (4, )),
        (((3, (3, 3)), (3, (3, 3))), (1, 1), 'VALID', (4, 4)),
        (((3, (3, 3)), (3, (3, 3))), (2, 2), 'SAME', (4, 4)),
    ])
    def test_build(self, filters, strides, padding, hidden_sizes):
        env = GarageEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        obs = env.reset()

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None) +
                                               policy.input_dim)
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run([policy.distribution.probs],
                                feed_dict={policy.model.input: [[obs]]})
        output2 = self.sess.run([dist_sym.probs],
                                feed_dict={state_input: [[obs]]})
        assert np.array_equal(output1, output2)

    def test_is_pickleable(self):
        env = GarageEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=((3, (32, 32)), ),
                                      strides=(1, ),
                                      padding='SAME',
                                      hidden_sizes=(4, ))

        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.compat.v1.variable_scope(
                'CategoricalCNNPolicy/CategoricalCNNModel', reuse=True):
            cnn_bias = tf.compat.v1.get_variable('CNNModel/cnn/h0/bias')
            bias = tf.compat.v1.get_variable('MLPModel/mlp/hidden_0/bias')

        cnn_bias.load(tf.ones_like(cnn_bias).eval())
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(policy.distribution.probs,
                                feed_dict={policy.model.input: [[obs]]})
        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(policy_pickled.distribution.probs,
                               feed_dict={policy_pickled.model.input: [[obs]]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (32, 32)), ), (1, ), 'VALID', (4, )),
        (((3, (32, 32)), (3, (64, 64))), (2, 2), 'SAME', (4, 4)),
    ])
    def test_clone(self, filters, strides, padding, hidden_sizes):
        env = GarageEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        policy_clone = policy.clone('CategoricalCNNPolicyClone')
        assert policy.env_spec == policy_clone.env_spec

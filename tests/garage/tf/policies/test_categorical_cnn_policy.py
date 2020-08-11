import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.policies import CategoricalCNNPolicy

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDictEnv, DummyDiscretePixelEnv


class TestCategoricalCNNPolicyWithModel(TfGraphTestCase):

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (3, 3)), ), (1, ), 'VALID', (4, )),
        (((3, (3, 3)), (3, (3, 3))), (1, 1), 'VALID', (4, 4)),
        (((3, (3, 3)), (3, (3, 3))), (2, 2), 'SAME', (4, 4)),
    ])
    def test_get_action(self, filters, strides, padding, hidden_sizes):
        env = GymEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        env.reset()
        obs = env.step(1).observation

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
        env = GymEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None) +
                                               policy.input_dim)
        dist_sym = policy.build(state_input, name='dist_sym').dist
        dist_sym2 = policy.build(state_input, name='dist_sym2').dist
        output1 = self.sess.run([dist_sym.probs],
                                feed_dict={state_input: [[obs]]})
        output2 = self.sess.run([dist_sym2.probs],
                                feed_dict={state_input: [[obs]]})
        assert np.array_equal(output1, output2)

    def test_is_pickleable(self):
        env = GymEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=((3, (32, 32)), ),
                                      strides=(1, ),
                                      padding='SAME',
                                      hidden_sizes=(4, ))

        env.reset()
        obs = env.step(1).observation

        with tf.compat.v1.variable_scope('CategoricalCNNPolicy', reuse=True):
            cnn_bias = tf.compat.v1.get_variable('CNNModel/cnn/h0/bias')
            bias = tf.compat.v1.get_variable('MLPModel/mlp/hidden_0/bias')

        cnn_bias.load(tf.ones_like(cnn_bias).eval())
        bias.load(tf.ones_like(bias).eval())

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None) +
                                               policy.input_dim)
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run(dist_sym.probs,
                                feed_dict={state_input: [[obs]]})
        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, None) +
                                                   policy.input_dim)
            dist_sym = policy_pickled.build(state_input, name='dist_sym').dist
            output2 = sess.run(dist_sym.probs,
                               feed_dict={state_input: [[obs]]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (32, 32)), ), (1, ), 'VALID', (4, )),
        (((3, (32, 32)), (3, (64, 64))), (2, 2), 'SAME', (4, 4)),
    ])
    def test_clone(self, filters, strides, padding, hidden_sizes):
        env = GymEnv(DummyDiscretePixelEnv())
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=filters,
                                      strides=strides,
                                      padding=padding,
                                      hidden_sizes=hidden_sizes)

        policy_clone = policy.clone('CategoricalCNNPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
        for cloned_param, param in zip(policy_clone.parameters.values(),
                                       policy.parameters.values()):
            assert np.array_equal(cloned_param, param)

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (32, 32)), ), (1, ), 'VALID', (4, )),
    ])
    def test_does_not_support_dict_obs_space(self, filters, strides, padding,
                                             hidden_sizes):
        """Test that policy raises error if passed a dict obs space."""
        env = GymEnv(DummyDictEnv(act_space_type='discrete'))
        with pytest.raises(ValueError):
            CategoricalCNNPolicy(env_spec=env.spec,
                                 filters=filters,
                                 strides=strides,
                                 padding=padding,
                                 hidden_sizes=hidden_sizes)


class TestCategoricalCNNPolicyImageObs(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = GymEnv(DummyDiscretePixelEnv(), is_image=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.env.reset()

    @pytest.mark.parametrize('filters, strides, padding, hidden_sizes', [
        (((3, (32, 32)), ), (1, ), 'VALID', (4, )),
    ])
    def test_obs_unflattened(self, filters, strides, padding, hidden_sizes):
        self.policy = CategoricalCNNPolicy(env_spec=self.env.spec,
                                           filters=filters,
                                           strides=strides,
                                           padding=padding,
                                           hidden_sizes=hidden_sizes)
        obs = self.env.observation_space.sample()
        action, _ = self.policy.get_action(
            self.env.observation_space.flatten(obs))
        self.env.step(action)

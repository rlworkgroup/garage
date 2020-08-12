import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.q_functions import DiscreteMLPDuelingQFunction

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestDiscreteMLPDuelingQFunction(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_sizes', [
        ((1, ), 1, (3, )),
        ((1, ), 1, (32, )),
        ((2, ), 2, (3, 3)),
        ((2, ), 2, (32, 32)),
    ])
    def test_get_action(self, obs_dim, action_dim, hidden_sizes):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = DiscreteMLPDuelingQFunction(env_spec=env.spec,
                                         hidden_sizes=hidden_sizes,
                                         hidden_w_init=tf.ones_initializer(),
                                         output_w_init=tf.ones_initializer())
        obs = np.full(obs_dim, 1)

        expected_output = np.full(action_dim,
                                  obs_dim[-1] * np.prod(hidden_sizes))

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert np.array_equal(outputs[0], expected_output)

        outputs = self.sess.run(qf.q_vals,
                                feed_dict={qf.input: [obs, obs, obs]})
        for output in outputs:
            assert np.array_equal(output, expected_output)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
    ])
    def test_output_shape(self, obs_dim, action_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = DiscreteMLPDuelingQFunction(env_spec=env.spec)
        env.reset()
        obs = env.step(1).observation

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert outputs.shape == (1, action_dim)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_build(self, obs_dim, action_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = DiscreteMLPDuelingQFunction(env_spec=env.spec)
        env.reset()
        obs = env.step(1).observation

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) + obs_dim)
        q_vals = qf.build(input_var, 'another')
        output2 = self.sess.run(q_vals, feed_dict={input_var: [obs]})

        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = DiscreteMLPDuelingQFunction(env_spec=env.spec)
        env.reset()
        obs = env.step(1).observation

        with tf.compat.v1.variable_scope('DiscreteMLPDuelingQFunction',
                                         reuse=True):
            bias = tf.compat.v1.get_variable('state_value/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            output2 = sess.run(qf_pickled.q_vals,
                               feed_dict={qf_pickled.input: [obs]})

        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_sizes', [
        ((1, ), 1, (3, )),
        ((2, ), 2, (32, )),
        ((1, 1), 1, (3, 3)),
        ((2, 2), 2, (32, 32)),
    ])
    def test_clone(self, obs_dim, action_dim, hidden_sizes):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        qf = DiscreteMLPDuelingQFunction(env_spec=env.spec,
                                         hidden_sizes=hidden_sizes)
        qf_clone = qf.clone('another_qf')
        assert qf_clone._hidden_sizes == qf._hidden_sizes

import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleMLPModel


class TestDiscreteMLPQFunction(TfGraphTestCase):
    @params(
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    )
    def test_get_action(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        expected_output = np.full(action_dim, 0.5)

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert np.array_equal(outputs[0], expected_output)

        outputs = self.sess.run(
            qf.q_vals, feed_dict={qf.input: [obs, obs, obs]})
        for output in outputs:
            assert np.array_equal(output, expected_output)

    @params(
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    )
    def test_output_shape(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert outputs.shape == (1, action_dim)

    @params(
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    )
    def test_get_qval_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        input_var = tf.placeholder(tf.float32, shape=(None, ) + obs_dim)
        q_vals = qf.get_qval_sym(input_var, 'another')
        output2 = self.sess.run(q_vals, feed_dict={input_var: [obs]})

        expected_output = np.full(action_dim, 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    @params(
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    )
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.variable_scope(
                'discrete_mlp_q_function/discrete_mlp_q_function', reuse=True):
            return_var = tf.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        h_data = pickle.dumps(qf)
        with tf.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            output2 = sess.run(
                qf_pickled.q_vals, feed_dict={qf_pickled.input: [obs]})

        assert np.array_equal(output1, output2)

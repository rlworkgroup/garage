import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleMLPModel


class TestDiscreteMLPQFunction(TfGraphTestCase):
    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
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

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_output_shape(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec, dueling=False)
        env.reset()
        obs, _, _, _ = env.step(1)

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert outputs.shape == (1, action_dim)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_output_shape_dueling(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPDuelingModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec, dueling=True)
        env.reset()
        obs, _, _, _ = env.step(1)

        outputs = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})
        assert outputs.shape == (1, action_dim)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_get_qval_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + obs_dim)
        q_vals = qf.get_qval_sym(input_var, 'another')
        output2 = self.sess.run(q_vals, feed_dict={input_var: [obs]})

        expected_output = np.full(action_dim, 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.compat.v1.variable_scope(
                'DiscreteMLPQFunction/SimpleMLPModel', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(qf.q_vals, feed_dict={qf.input: [obs]})

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            output2 = sess.run(
                qf_pickled.q_vals, feed_dict={qf_pickled.input: [obs]})

        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_sizes', [
        ((1, ), 1, (3, )),
        ((2, ), 2, (32, )),
        ((1, 1), 1, (3, 3)),
        ((2, 2), 2, (32, 32)),
    ])
    def test_clone(self, obs_dim, action_dim, hidden_sizes):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.q_functions.'
                         'discrete_mlp_q_function.MLPModel'),
                        new=SimpleMLPModel):
            qf = DiscreteMLPQFunction(
                env_spec=env.spec, hidden_sizes=hidden_sizes)
        qf_clone = qf.clone('another_qf')
        assert qf_clone._hidden_sizes == qf._hidden_sizes

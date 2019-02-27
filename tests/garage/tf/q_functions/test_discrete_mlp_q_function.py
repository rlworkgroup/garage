"""This script creates a unittest that tests discrete mlp q-function."""
import pickle

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestDiscreteMLPQFunction(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.data = np.ones((2, 1))
        self.env = TfEnv(DummyDiscreteEnv())
        self.qf = DiscreteMLPQFunction(self.env.spec)

    def test_discrete_mlp_q_function(self):
        output1 = self.sess.run(
            self.qf.model.networks['default'].outputs,
            feed_dict={self.qf.model.networks['default'].input: self.data})
        assert output1.shape == (2, self.env.action_space.n)

    def test_discrete_mlp_q_function_is_rebuilt_output_same(self):
        output1 = self.sess.run(
            self.qf.model.networks['default'].outputs,
            feed_dict={self.qf.model.networks['default'].input: self.data})

        input_var = tf.placeholder(tf.float32, shape=(None, 1))
        q_vals = self.qf.get_qval_sym(input_var, "another")
        output2 = self.sess.run(q_vals, feed_dict={input_var: self.data})

        assert np.array_equal(output1, output2)

    def test_discrete_mlp_q_function_is_pickleable(self):
        output1 = self.sess.run(
            self.qf.model.networks['default'].outputs,
            feed_dict={self.qf.model.networks['default'].input: self.data})
        h_data = pickle.dumps(self.qf)

        with tf.Session(graph=tf.Graph()) as sess:
            qf_pickled = pickle.loads(h_data)
            input_var = tf.placeholder(tf.float32, shape=(None, 1))
            q_vals = qf_pickled.get_qval_sym(input_var, "another")
            output2 = sess.run(q_vals, feed_dict={input_var: self.data})

        assert np.array_equal(output1, output2)

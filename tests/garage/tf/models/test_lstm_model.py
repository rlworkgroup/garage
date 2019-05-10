import pickle

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.models import LSTMModel
from tests.fixtures import TfGraphTestCase


class TestLSTMModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 1
        self.time_step = 1
        self.feature_shape = 2
        self.output_dim = 1

        self.obs_inputs = np.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
        self.obs_input = np.full((self.batch_size, self.feature_shape), 1.)

        self._input_var = tf.placeholder(
            tf.float32, shape=(None, None, self.feature_shape), name='input')
        self._step_input_var = tf.placeholder(
            tf.float32, shape=(None, self.feature_shape), name='input')

    @params((1, 1), (1, 2), (3, 3))
    def test_output_values(self, output_dim, hidden_dim):
        model = LSTMModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=None,
            recurrent_nonlinearity=None,
            hidden_w_init=tf.constant_initializer(1),
            recurrent_w_init=tf.constant_initializer(1),
            output_w_init=tf.constant_initializer(1))

        step_hidden_var = tf.placeholder(
            shape=(self.batch_size, hidden_dim),
            name='step_hidden',
            dtype=tf.float32)
        step_cell_var = tf.placeholder(
            shape=(self.batch_size, hidden_dim),
            name='step_cell',
            dtype=tf.float32)

        outputs = model.build(self._input_var, self._step_input_var,
                              step_hidden_var, step_cell_var)
        output = self.sess.run(
            outputs[0], feed_dict={self._input_var: self.obs_inputs})
        expected_output = np.full(
            [self.batch_size, self.time_step, output_dim], hidden_dim * 8)
        assert np.array_equal(output, expected_output)

    def test_is_pickleable(self):
        model = LSTMModel(output_dim=1, hidden_dim=1)
        step_hidden_var = tf.placeholder(
            shape=(self.batch_size, 1), name='step_hidden', dtype=tf.float32)
        step_cell_var = tf.placeholder(
            shape=(self.batch_size, 1), name='step_cell', dtype=tf.float32)
        outputs = model.build(self._input_var, self._step_input_var,
                              step_hidden_var, step_cell_var)

        # assign bias to all one
        with tf.variable_scope('LSTMModel/lstm', reuse=True):
            init_hidden = tf.get_variable('initial_hidden')

        init_hidden.load(tf.ones_like(init_hidden).eval())

        hidden = np.zeros((self.batch_size, 1))
        cell = np.zeros((self.batch_size, 1))

        outputs1 = self.sess.run(
            outputs[0], feed_dict={self._input_var: self.obs_inputs})
        output1 = self.sess.run(
            outputs[1:4],
            feed_dict={
                self._step_input_var: self.obs_input,
                step_hidden_var: hidden,
                step_cell_var: cell
            })

        h = pickle.dumps(model)
        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(h)

            input_var = tf.placeholder(
                tf.float32,
                shape=(None, None, self.feature_shape),
                name='input')
            step_input_var = tf.placeholder(
                tf.float32, shape=(None, self.feature_shape), name='input')
            step_hidden_var = tf.placeholder(
                shape=(self.batch_size, 1),
                name='initial_hidden',
                dtype=tf.float32)
            step_cell_var = tf.placeholder(
                shape=(self.batch_size, 1),
                name='initial_cell',
                dtype=tf.float32)

            outputs = model_pickled.build(input_var, step_input_var,
                                          step_hidden_var, step_cell_var)
            outputs2 = sess.run(
                outputs[0], feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                outputs[1:4],
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden,
                    step_cell_var: cell
                })
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)

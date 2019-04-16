import numpy as np
import tensorflow as tf

from garage.tf.core.lstm import lstm
from tests.fixtures import TfGraphTestCase


class TestLSTM(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 1
        self.time_step = 5
        self.feature_shape = 3
        self.hidden_dim = 2
        self.output_dim = 1

        self.obs_inputs = np.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
        self.obs_input = np.full((self.batch_size, self.feature_shape), 1.)

        self._input_var = tf.placeholder(
            tf.float32, shape=(None, None, self.feature_shape), name='input')
        self._step_input_var = tf.placeholder(
            tf.float32, shape=(None, self.feature_shape), name='input')
        self._step_hidden_var = tf.placeholder(
            shape=(self.batch_size, self.hidden_dim),
            name='initial_hidden',
            dtype=tf.float32)
        self._step_cell_var = tf.placeholder(
            shape=(self.batch_size, self.hidden_dim),
            name='initial_cell',
            dtype=tf.float32)

        self.lstm_cell = tf.keras.layers.LSTMCell(
            units=self.hidden_dim,
            activation=tf.nn.tanh,
            kernel_initializer=tf.constant_initializer(1),
            recurrent_activation=tf.nn.sigmoid,
            recurrent_initializer=tf.constant_initializer(1),
            name='lstm_layer')
        self.output_nonlinearity = tf.keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.variable_scope('LSTM'):
            self.lstm = lstm(
                all_input_var=self._input_var,
                name='lstm',
                lstm_cell=self.lstm_cell,
                step_input_var=self._step_input_var,
                step_hidden_var=self._step_hidden_var,
                step_cell_var=self._step_cell_var,
                output_nonlinearity_layer=self.output_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

    def test_output_same_as_rnn(self):
        # Create a RNN and compute the entire outputs
        rnn_layer = tf.keras.layers.RNN(
            cell=self.lstm_cell, return_sequences=True, return_state=True)

        # Set initial state to all 0s
        hidden_var = tf.get_variable(
            name='initial_hidden',
            shape=(self.batch_size, self.hidden_dim),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)
        cell_var = tf.get_variable(
            name='initial_cell',
            shape=(self.batch_size, self.hidden_dim),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)
        outputs, hiddens, cells = rnn_layer(
            self._input_var, initial_state=[hidden_var, cell_var])
        outputs = self.output_nonlinearity(outputs)

        self.sess.run(tf.global_variables_initializer())

        outputs, hiddens, cells = self.sess.run(
            [outputs, hiddens, cells],
            feed_dict={self._input_var: self.obs_inputs})

        # Compute output by doing t step() on the lstm cell
        # Set initial state to all 0s
        hidden = np.zeros((self.batch_size, self.hidden_dim))
        cell = np.zeros((self.batch_size, self.hidden_dim))
        for i in range(self.time_step):
            output, hidden, cell = self.sess.run(
                self.lstm[1:4],
                feed_dict={
                    self._step_input_var: self.obs_input,
                    self._step_hidden_var: hidden,
                    self._step_cell_var: cell
                })
            # The output from i-th timestep
            assert np.array_equal(output, outputs[:, i, :])
        assert np.array_equal(hidden, hiddens)
        assert np.array_equal(cell, cells)

        # Also the full output from lstm
        full_outputs = self.sess.run(
            self.lstm[0], feed_dict={self._input_var: self.obs_inputs})
        assert np.array_equal(outputs, full_outputs)

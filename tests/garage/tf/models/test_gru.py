import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models.gru import gru

from tests.fixtures import TfGraphTestCase
from tests.helpers import recurrent_step_gru


class TestGRU(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.batch_size = 2
        self.hidden_dim = 2

        self.step_hidden_var = tf.compat.v1.placeholder(
            shape=(self.batch_size, self.hidden_dim),
            name='initial_hidden',
            dtype=tf.float32)

        self.gru_cell = tf.keras.layers.GRUCell(
            units=self.hidden_dim,
            activation=tf.nn.tanh,
            kernel_initializer=tf.constant_initializer(1),
            recurrent_activation=tf.nn.sigmoid,
            recurrent_initializer=tf.constant_initializer(1),
            name='lstm_layer')

    # yapf: disable
    @pytest.mark.parametrize(
        'time_step, input_dim, output_dim, hidden_init', [
            (1, 1, 1, 0),
            (1, 1, 3, 0),
            (1, 3, 1, 0),
            (3, 1, 1, 0),
            (3, 3, 1, 0),
            (3, 3, 3, 0),
            (1, 1, 1, 0.5),
            (1, 1, 3, 0.5),
            (1, 3, 1, 0.5),
            (3, 1, 1, 0.5),
            (3, 3, 1, 0.5),
            (3, 3, 3, 0.5),
        ])
    # yapf: enable
    def test_output_shapes(self, time_step, input_dim, output_dim,
                           hidden_init):
        obs_inputs = np.full((self.batch_size, time_step, input_dim), 1.)
        obs_input = np.full((self.batch_size, input_dim), 1.)

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, input_dim),
                                             name='input')
        step_input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, input_dim),
                                                  name='step_input')
        output_nonlinearity = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.compat.v1.variable_scope('GRU'):
            self.gru = gru(
                all_input_var=input_var,
                name='gru',
                gru_cell=self.gru_cell,
                step_input_var=step_input_var,
                step_hidden_var=self.step_hidden_var,
                hidden_state_init=tf.constant_initializer(hidden_init),
                output_nonlinearity_layer=output_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Compute output by doing t step() on the gru cell
        outputs_t, output_t, h_t, hidden_init = self.gru
        hidden = np.full((self.batch_size, self.hidden_dim),
                         hidden_init.eval())

        for _ in range(time_step):
            output, hidden = self.sess.run([output_t, h_t],
                                           feed_dict={
                                               step_input_var: obs_input,
                                               self.step_hidden_var: hidden,
                                           })  # noqa: E126
            assert output.shape == (self.batch_size, output_dim)
            assert hidden.shape == (self.batch_size, self.hidden_dim)

        full_output = self.sess.run(outputs_t,
                                    feed_dict={input_var: obs_inputs})

        assert full_output.shape == (self.batch_size, time_step, output_dim)

    # yapf: disable
    @pytest.mark.parametrize(
        'time_step, input_dim, output_dim, hidden_init', [
            (1, 1, 1, 0),
            (1, 1, 3, 0),
            (1, 3, 1, 0),
            (3, 1, 1, 0),
            (3, 3, 1, 0),
            (3, 3, 3, 0),
            (1, 1, 1, 0.5),
            (1, 1, 3, 0.5),
            (1, 3, 1, 0.5),
            (3, 1, 1, 0.5),
            (3, 3, 1, 0.5),
            (3, 3, 3, 0.5),
        ])
    # yapf: enable
    def test_output_value(self, time_step, input_dim, output_dim, hidden_init):
        obs_inputs = np.full((self.batch_size, time_step, input_dim), 1.)
        obs_input = np.full((self.batch_size, input_dim), 1.)

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, input_dim),
                                             name='input')
        step_input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, input_dim),
                                                  name='step_input')
        output_nonlinearity = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.compat.v1.variable_scope('GRU'):
            self.gru = gru(
                all_input_var=input_var,
                name='gru',
                gru_cell=self.gru_cell,
                step_input_var=step_input_var,
                step_hidden_var=self.step_hidden_var,
                hidden_state_init=tf.constant_initializer(hidden_init),
                output_nonlinearity_layer=output_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Compute output by doing t step() on the gru cell
        outputs_t, output_t, h_t, hidden_init = self.gru
        hidden1 = hidden2 = np.full((self.batch_size, self.hidden_dim),
                                    hidden_init.eval())

        for i in range(time_step):
            output1, hidden1 = self.sess.run([output_t, h_t],
                                             feed_dict={
                                                 step_input_var: obs_input,
                                                 self.step_hidden_var: hidden1
                                             })  # noqa: E126

            hidden2 = recurrent_step_gru(input_val=obs_input,
                                         num_units=self.hidden_dim,
                                         step_hidden=hidden2,
                                         w_x_init=1.,
                                         w_h_init=1.,
                                         b_init=0.,
                                         nonlinearity=np.tanh,
                                         gate_nonlinearity=lambda x: 1. /
                                         (1. + np.exp(-x)))

            output_nonlinearity = np.full(
                (np.prod(hidden2.shape[1:]), output_dim), 1.)
            output2 = np.matmul(hidden2, output_nonlinearity)

            assert np.allclose(output1, output2)
            assert np.allclose(hidden1, hidden2)

        full_output1 = self.sess.run(outputs_t,
                                     feed_dict={input_var: obs_inputs})

        hidden2 = np.full((self.batch_size, self.hidden_dim),
                          hidden_init.eval())
        stack_hidden = None
        for i in range(time_step):
            hidden2 = recurrent_step_gru(input_val=obs_inputs[:, i, :],
                                         num_units=self.hidden_dim,
                                         step_hidden=hidden2,
                                         w_x_init=1.,
                                         w_h_init=1.,
                                         b_init=0.,
                                         nonlinearity=np.tanh,
                                         gate_nonlinearity=lambda x: 1. /
                                         (1. + np.exp(-x)))
            if stack_hidden is None:
                stack_hidden = hidden2[:, np.newaxis, :]
            else:
                stack_hidden = np.concatenate(
                    (stack_hidden, hidden2[:, np.newaxis, :]), axis=1)
        output_nonlinearity = np.full((np.prod(hidden2.shape[1:]), output_dim),
                                      1.)
        full_output2 = np.matmul(stack_hidden, output_nonlinearity)
        assert np.allclose(full_output1, full_output2)

    # yapf: disable
    @pytest.mark.parametrize('time_step, input_dim, output_dim', [
        (1, 1, 1),
        (1, 1, 3),
        (1, 3, 1),
        (3, 1, 1),
        (3, 3, 1),
        (3, 3, 3),
    ])
    # yapf: enable
    def test_output_value_trainable_hidden_and_cell(self, time_step, input_dim,
                                                    output_dim):
        obs_inputs = np.full((self.batch_size, time_step, input_dim), 1.)
        obs_input = np.full((self.batch_size, input_dim), 1.)

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, input_dim),
                                             name='input')
        step_input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, input_dim),
                                                  name='step_input')
        output_nonlinearity = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.compat.v1.variable_scope('GRU'):
            self.gru = gru(all_input_var=input_var,
                           name='gru',
                           gru_cell=self.gru_cell,
                           step_input_var=step_input_var,
                           step_hidden_var=self.step_hidden_var,
                           hidden_state_init_trainable=True,
                           output_nonlinearity_layer=output_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Compute output by doing t step() on the gru cell
        outputs_t, output_t, h_t, hidden_init = self.gru
        hidden = np.full((self.batch_size, self.hidden_dim),
                         hidden_init.eval())

        _, hidden = self.sess.run([output_t, h_t],
                                  feed_dict={
                                      step_input_var: obs_input,
                                      self.step_hidden_var: hidden,
                                  })  # noqa: E126
        with tf.compat.v1.variable_scope('GRU/gru', reuse=True):
            hidden_init_var = tf.compat.v1.get_variable(name='initial_hidden')
            assert hidden_init_var in tf.compat.v1.trainable_variables()

        full_output1 = self.sess.run(outputs_t,
                                     feed_dict={input_var: obs_inputs})

        hidden2 = np.full((self.batch_size, self.hidden_dim),
                          hidden_init.eval())
        stack_hidden = None
        for i in range(time_step):
            hidden2 = recurrent_step_gru(input_val=obs_inputs[:, i, :],
                                         num_units=self.hidden_dim,
                                         step_hidden=hidden2,
                                         w_x_init=1.,
                                         w_h_init=1.,
                                         b_init=0.,
                                         nonlinearity=np.tanh,
                                         gate_nonlinearity=lambda x: 1. /
                                         (1. + np.exp(-x)))
            if stack_hidden is None:
                stack_hidden = hidden2[:, np.newaxis, :]
            else:
                stack_hidden = np.concatenate(
                    (stack_hidden, hidden2[:, np.newaxis, :]), axis=1)
        output_nonlinearity = np.full((np.prod(hidden2.shape[1:]), output_dim),
                                      1.)
        full_output2 = np.matmul(stack_hidden, output_nonlinearity)
        assert np.allclose(full_output1, full_output2)

    def test_gradient_paths(self):
        time_step = 3
        input_dim = 2
        output_dim = 4
        obs_inputs = np.full((self.batch_size, time_step, input_dim), 1.)
        obs_input = np.full((self.batch_size, input_dim), 1.)

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, input_dim),
                                             name='input')
        step_input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, input_dim),
                                                  name='step_input')
        output_nonlinearity = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.compat.v1.variable_scope('GRU'):
            self.gru = gru(all_input_var=input_var,
                           name='gru',
                           gru_cell=self.gru_cell,
                           step_input_var=step_input_var,
                           step_hidden_var=self.step_hidden_var,
                           output_nonlinearity_layer=output_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Compute output by doing t step() on the gru cell
        outputs_t, output_t, h_t, hidden_init = self.gru
        hidden = np.full((self.batch_size, self.hidden_dim),
                         hidden_init.eval())

        grads_step_o_i = tf.gradients(output_t, step_input_var)
        grads_step_o_h = tf.gradients(output_t, self.step_hidden_var)
        grads_step_h = tf.gradients(h_t, step_input_var)

        self.sess.run([grads_step_o_i, grads_step_o_h, grads_step_h],
                      feed_dict={
                          step_input_var: obs_input,
                          self.step_hidden_var: hidden,
                      })  # noqa: E126

        grads_step_o_i = tf.gradients(outputs_t, step_input_var)
        grads_step_o_h = tf.gradients(outputs_t, self.step_hidden_var)
        grads_step_h = tf.gradients(h_t, input_var)

        # No gradient flow
        with pytest.raises(TypeError):
            self.sess.run(grads_step_o_i,
                          feed_dict={
                              step_input_var: obs_input,
                              self.step_hidden_var: hidden,
                          })
        with pytest.raises(TypeError):
            self.sess.run(grads_step_o_h,
                          feed_dict={
                              step_input_var: obs_input,
                              self.step_hidden_var: hidden,
                          })
        with pytest.raises(TypeError):
            self.sess.run(grads_step_h, feed_dict={input_var: obs_inputs})

    # yapf: disable
    @pytest.mark.parametrize(
        'time_step, input_dim, output_dim, hidden_init', [
            (1, 1, 1, 0),
            (1, 1, 3, 0),
            (1, 3, 1, 0),
            (3, 1, 1, 0),
            (3, 3, 1, 0),
            (3, 3, 3, 0),
            (1, 1, 1, 0.5),
            (1, 1, 3, 0.5),
            (1, 3, 1, 0.5),
            (3, 1, 1, 0.5),
            (3, 3, 1, 0.5),
            (3, 3, 3, 0.5),
        ])
    # yapf: enable
    def test_output_same_as_rnn(self, time_step, input_dim, output_dim,
                                hidden_init):
        obs_inputs = np.full((self.batch_size, time_step, input_dim), 1.)
        obs_input = np.full((self.batch_size, input_dim), 1.)

        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, input_dim),
                                             name='input')
        step_input_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, input_dim),
                                                  name='step_input')
        output_nonlinearity = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_initializer=tf.constant_initializer(1))
        with tf.compat.v1.variable_scope('GRU'):
            self.gru = gru(
                all_input_var=input_var,
                name='gru',
                gru_cell=self.gru_cell,
                step_input_var=step_input_var,
                step_hidden_var=self.step_hidden_var,
                hidden_state_init=tf.constant_initializer(hidden_init),
                output_nonlinearity_layer=output_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Create a RNN and compute the entire outputs
        rnn_layer = tf.keras.layers.RNN(cell=self.gru_cell,
                                        return_sequences=True,
                                        return_state=True)

        # Set initial state to all 0s
        hidden_var = tf.compat.v1.get_variable(
            name='initial_hidden',
            shape=(self.batch_size, self.hidden_dim),
            initializer=tf.constant_initializer(hidden_init),
            trainable=False,
            dtype=tf.float32)
        outputs, hiddens = rnn_layer(input_var, initial_state=[hidden_var])
        outputs = output_nonlinearity(outputs)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        outputs, hiddens = self.sess.run([outputs, hiddens],
                                         feed_dict={input_var: obs_inputs})

        # Compute output by doing t step() on the gru cell
        hidden = np.full((self.batch_size, self.hidden_dim), hidden_init)
        _, output_t, hidden_t, _ = self.gru
        for i in range(time_step):
            output, hidden = self.sess.run([output_t, hidden_t],
                                           feed_dict={
                                               step_input_var: obs_input,
                                               self.step_hidden_var: hidden,
                                           })  # noqa: E126
            # The output from i-th timestep
            assert np.array_equal(output, outputs[:, i, :])
        assert np.array_equal(hidden, hiddens)

        # Also the full output from lstm
        full_outputs = self.sess.run(self.gru[0],
                                     feed_dict={input_var: obs_inputs})
        assert np.array_equal(outputs, full_outputs)

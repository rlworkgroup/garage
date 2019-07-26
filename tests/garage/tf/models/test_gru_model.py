import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import GRUModel
from tests.fixtures import TfGraphTestCase


class TestGRUModel(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        self.batch_size = 1
        self.time_step = 1
        self.feature_shape = 2
        self.output_dim = 1

        self.obs_inputs = np.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
        self.obs_input = np.full((self.batch_size, self.feature_shape), 1.)

        self.input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, None, self.feature_shape), name='input')
        self.step_input_var = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.feature_shape), name='input')

    @pytest.mark.parametrize('output_dim, hidden_dim', [(1, 1), (1, 2),
                                                        (3, 3)])
    def test_output_values(self, output_dim, hidden_dim):
        model = GRUModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=None,
            recurrent_nonlinearity=None,
            hidden_w_init=tf.constant_initializer(1),
            recurrent_w_init=tf.constant_initializer(1),
            output_w_init=tf.constant_initializer(1))

        step_hidden_var = tf.compat.v1.placeholder(
            shape=(self.batch_size, hidden_dim),
            name='step_hidden',
            dtype=tf.float32)

        outputs = model.build(self.input_var, self.step_input_var,
                              step_hidden_var)
        output = self.sess.run(
            outputs[0], feed_dict={self.input_var: self.obs_inputs})
        expected_output = np.full(
            [self.batch_size, self.time_step, output_dim], hidden_dim * -2)
        assert np.array_equal(output, expected_output)

    def test_is_pickleable(self):
        model = GRUModel(output_dim=1, hidden_dim=1)
        step_hidden_var = tf.compat.v1.placeholder(
            shape=(self.batch_size, 1), name='step_hidden', dtype=tf.float32)
        model.build(self.input_var, self.step_input_var, step_hidden_var)

        # assign bias to all one
        with tf.compat.v1.variable_scope('GRUModel/gru', reuse=True):
            init_hidden = tf.compat.v1.get_variable('initial_hidden')

        init_hidden.load(tf.ones_like(init_hidden).eval())

        hidden = np.zeros((self.batch_size, 1))

        outputs1 = self.sess.run(
            model.networks['default'].all_output,
            feed_dict={self.input_var: self.obs_inputs})
        output1 = self.sess.run(
            [
                model.networks['default'].step_output,
                model.networks['default'].step_hidden
            ],
            # yapf: disable
            feed_dict={
                self.step_input_var: self.obs_input,
                step_hidden_var: hidden
            })
        # yapf: enable
        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(h)

            input_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, None, self.feature_shape),
                name='input')
            step_input_var = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self.feature_shape), name='input')
            step_hidden_var = tf.compat.v1.placeholder(
                shape=(self.batch_size, 1),
                name='initial_hidden',
                dtype=tf.float32)

            model_pickled.build(input_var, step_input_var, step_hidden_var)
            outputs2 = sess.run(
                model_pickled.networks['default'].all_output,
                feed_dict={input_var: self.obs_inputs})
            output2 = sess.run(
                [
                    model_pickled.networks['default'].step_output,
                    model_pickled.networks['default'].step_hidden
                ],
                # yapf: disable
                feed_dict={
                    step_input_var: self.obs_input,
                    step_hidden_var: hidden
                })
            # yapf: enable
            assert np.array_equal(outputs1, outputs2)
            assert np.array_equal(output1, output2)

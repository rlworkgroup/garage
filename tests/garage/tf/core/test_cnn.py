import numpy as np
import tensorflow as tf

from garage.tf.core.cnn import cnn
from garage.tf.core.cnn import cnn_with_max_pooling
from tests.fixtures import TfGraphTestCase
from tests.helpers import convolve
from tests.helpers import max_pooling


class TestCNN(TfGraphTestCase):
    def setUp(self):
        super(TestCNN, self).setUp()
        self.batch_size = 5
        self.input_width = 10
        self.input_height = 10
        self.obs_input = np.ones((self.batch_size, self.input_width,
                                  self.input_height, 3))

        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self._input_ph = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape, name="input")

        self._output_shape = 2
        self.filter_sizes = (3, 3)
        self.in_channels = (3, 32)
        self.out_channels = (32, 64)

        self.hidden_nonlinearity = tf.nn.relu

    def test_output_shape(self):
        with tf.variable_scope("CNN"):
            self.cnn = cnn(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=self.filter_sizes,
                num_filters=self.out_channels,
                stride=1,
                name="cnn1",
                padding="VALID",
                hidden_w_init=tf.constant_initializer(1),
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(
            self.cnn, feed_dict={self._input_ph: self.obs_input})
        assert result.shape[1] == self._output_shape

    def test_output_with_identity_filter(self):
        stride = 1
        with tf.variable_scope("CNN"):
            self.cnn = cnn(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=self.filter_sizes,
                num_filters=self.out_channels,
                stride=stride,
                name="cnn1",
                padding="VALID",
                hidden_w_init=tf.constant_initializer(1),
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(
            self.cnn, feed_dict={self._input_ph: self.obs_input})

        # get weight values
        with tf.variable_scope("CNN", reuse=True):
            out_w = tf.get_variable("cnn1/output/kernel")
            out_b = tf.get_variable("cnn1/output/bias")

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_size, in_channel in zip(self.filter_sizes,
                                           self.in_channels):
            filter_sum *= filter_size * filter_size * in_channel

        # input shape 10 * 10 * 3
        # after two conv layer, we have filter
        # with shape 6 * 6 * 32 (last channel)

        # flatten
        h_out = np.full(
            (self.batch_size, 6 * 6 * self.out_channels[-1]),
            filter_sum,
            dtype=np.float32)
        # pass to a dense layer
        dense_in = tf.matmul(h_out, out_w) + out_b
        np.testing.assert_array_equal(dense_in.eval(), result)

    def test_output_with_random_filter(self):
        stride = 1
        # Build a cnn with random filter weights
        with tf.variable_scope("CNN"):
            self.cnn2 = cnn(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=self.filter_sizes,
                num_filters=self.out_channels,
                stride=stride,
                name="cnn1",
                padding="VALID",
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(
            self.cnn2, feed_dict={self._input_ph: self.obs_input})

        # get weight values
        with tf.variable_scope("CNN", reuse=True):
            h0_w = tf.get_variable("cnn1/h0/weight").eval()
            h0_b = tf.get_variable("cnn1/h0/bias").eval()
            h1_w = tf.get_variable("cnn1/h1/weight").eval()
            h1_b = tf.get_variable("cnn1/h1/bias").eval()
            out_w = tf.get_variable("cnn1/output/kernel")
            out_b = tf.get_variable("cnn1/output/bias")

        filter_weights = (h0_w, h1_w)
        filter_bias = (h0_b, h1_b)

        # convolution according to TensorFlow's approach
        input_val = convolve(
            _input=self.obs_input,
            filter_weights=filter_weights,
            filter_bias=filter_bias,
            stride=stride,
            filter_sizes=self.filter_sizes,
            in_channels=self.in_channels,
            hidden_nonlinearity=self.hidden_nonlinearity)

        # flatten
        dense_in = input_val.reshape((self.batch_size, -1)).astype(np.float32)
        # pass to a dense layer
        dense_out = tf.matmul(dense_in, out_w) + out_b
        np.testing.assert_array_almost_equal(dense_out.eval(), result)

    def test_output_with_max_pooling(self):
        stride = 1
        pool_shape = 2
        pool_stride = 2
        # Build a cnn with random filter weights
        with tf.variable_scope("CNN"):
            self.cnn2 = cnn_with_max_pooling(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=self.filter_sizes,
                num_filters=self.out_channels,
                stride=stride,
                name="cnn1",
                pool_shape=(pool_shape, pool_shape),
                pool_stride=(pool_stride, pool_stride),
                padding="VALID",
                hidden_w_init=tf.constant_initializer(1),
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(
            self.cnn2, feed_dict={self._input_ph: self.obs_input})

        # get weight values
        with tf.variable_scope("CNN", reuse=True):
            h0_w = tf.get_variable("cnn1/h0/weight").eval()
            h0_b = tf.get_variable("cnn1/h0/bias").eval()
            h1_w = tf.get_variable("cnn1/h1/weight").eval()
            h1_b = tf.get_variable("cnn1/h1/bias").eval()
            out_w = tf.get_variable("cnn1/output/kernel")
            out_b = tf.get_variable("cnn1/output/bias")

        filter_weights = (h0_w, h1_w)
        filter_bias = (h0_b, h1_b)

        input_val = self.obs_input

        # convolution according to TensorFlow's approach
        # and perform max pooling on each layer
        for filter_size, filter_weight, _filter_bias, in_channel in zip(
                self.filter_sizes, filter_weights, filter_bias,
                self.in_channels):
            input_val = convolve(
                _input=input_val,
                filter_weights=(filter_weight, ),
                filter_bias=(_filter_bias, ),
                stride=stride,
                filter_sizes=(filter_size, ),
                in_channels=(in_channel, ),
                hidden_nonlinearity=self.hidden_nonlinearity)

            # max pooling
            input_val = max_pooling(
                _input=input_val,
                pool_shape=pool_shape,
                pool_stride=pool_stride)

        # flatten
        dense_in = input_val.reshape((self.batch_size, -1)).astype(np.float32)
        # pass to a dense layer
        dense_out = tf.matmul(dense_in, out_w) + out_b
        np.testing.assert_array_equal(dense_out.eval(), result)

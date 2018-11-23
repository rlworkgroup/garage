import numpy as np
import tensorflow as tf

from garage.tf.core.cnn import cnn
from garage.tf.core.cnn import cnn_with_max_pooling
from tests.fixtures import TfGraphTestCase


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
        self.filter_sizes = (3, )
        self.in_channels = (3, )
        self.out_channels = (32, )

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
        # after conv layer, we have filter with shape 8 * 8 * 32 (last channel)
        # flatten
        h_out = np.full(
            (self.batch_size, 8 * 8 * self.out_channels[-1]),
            filter_sum,
            dtype=np.float32)
        # pass to a dense layer
        dense_in = tf.matmul(h_out, out_w) + out_b
        np.testing.assert_array_equal(dense_in.eval(), result)

    def convolve(self, _input, filter_weights, filter_biass, stride):
        in_width = self.input_width
        in_height = self.input_height

        for filter_size, in_shape, filter_weight, filter_bias in zip(
                self.filter_sizes, self.in_channels, filter_weights,
                filter_biass):
            out_width = int((in_width - filter_size) / stride) + 1
            out_height = int((in_height - filter_size) / stride) + 1
            flatten_filter_size = filter_size * filter_size * in_shape
            reshape_filter = filter_weight.reshape(flatten_filter_size, -1)
            image_vector = np.empty((self.batch_size, out_width, out_height,
                                     flatten_filter_size))
            for batch in range(self.batch_size):
                for w in range(out_width):
                    for h in range(out_height):
                        sliding_window = np.empty((filter_size, filter_size,
                                                   in_shape))
                        for dw in range(filter_size):
                            for dh in range(filter_size):
                                for in_c in range(in_shape):
                                    sliding_window[dw][dh][in_c] = _input[
                                        batch][w + dw][h + dh][in_c]
                        image_vector[batch][w][h] = sliding_window.flatten()
            _input = np.dot(image_vector, reshape_filter) + filter_bias
            _input = self.hidden_nonlinearity(_input).eval()

            in_width = out_width
            in_height = out_height

        return _input

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
            out_w = tf.get_variable("cnn1/output/kernel")
            out_b = tf.get_variable("cnn1/output/bias")

        filter_weights = (h0_w, )
        filter_biass = (h0_b, )

        # convolution according to TensorFlow's approach
        input_val = self.convolve(
            _input=self.obs_input,
            filter_weights=filter_weights,
            filter_biass=filter_biass,
            stride=stride)

        # flatten
        input_val = input_val.reshape((self.batch_size, -1)).astype(np.float32)
        # pass to a dense layer
        dense_in = tf.matmul(input_val, out_w) + out_b
        np.testing.assert_array_almost_equal(dense_in.eval(), result)

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
            out_w = tf.get_variable("cnn1/output/kernel")
            out_b = tf.get_variable("cnn1/output/bias")

        filter_weights = (h0_w, )
        filter_biass = (h0_b, )

        # convolution according to TensorFlow's approach
        input_val = self.convolve(
            _input=self.obs_input,
            filter_weights=filter_weights,
            filter_biass=filter_biass,
            stride=stride)

        # max pooling
        results = np.empty((self.batch_size,
                            int(input_val.shape[1] / pool_shape),
                            int(input_val.shape[2] / pool_shape),
                            input_val.shape[3]))
        for b in range(self.batch_size):
            for i, row in enumerate(range(0, input_val.shape[1], pool_stride)):
                for j, col in enumerate(
                        range(0, input_val.shape[2], pool_stride)):
                    for k in range(input_val.shape[3]):
                        local_max = input_val[b, col:col + pool_shape, row:
                                              row + pool_shape, k]
                        results[b][i][j][k] = np.max(local_max)

        # flatten
        results = results.reshape((self.batch_size, -1)).astype(np.float32)
        # pass to a dense layer
        dense_in = tf.matmul(results, out_w) + out_b
        np.testing.assert_array_equal(dense_in.eval(), result)

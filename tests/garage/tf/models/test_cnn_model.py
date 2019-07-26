import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import CNNModel
from garage.tf.models import CNNModelWithMaxPooling
from tests.fixtures import TfGraphTestCase


class TestCNNModel(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        self.batch_size = 5
        self.input_width = 10
        self.input_height = 10
        self.obs_input = np.ones((self.batch_size, self.input_width,
                                  self.input_height, 3))
        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self._input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, ) + input_shape, name='input')

    # yapf: disable
    @pytest.mark.parametrize('filter_sizes, in_channels, out_channels, '
                             'strides', [
        ((1,), (3,), (32,), (1,)),  # noqa: E122
        ((3,), (3,), (32,), (1,)),
        ((3,), (3,), (32,), (2,)),
        ((1, 1), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (2, 2)),
    ])
    # yapf: enable
    def test_output_value(self, filter_sizes, in_channels, out_channels,
                          strides):
        model = CNNModel(
            filter_dims=filter_sizes,
            num_filters=out_channels,
            strides=strides,
            name='cnn_model',
            padding='VALID',
            hidden_w_init=tf.constant_initializer(1),
            hidden_nonlinearity=None)

        outputs = model.build(self._input_ph)
        output = self.sess.run(
            outputs, feed_dict={self._input_ph: self.obs_input})

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_size, in_channel in zip(filter_sizes, in_channels):
            filter_sum *= filter_size * filter_size * in_channel

        current_size = self.input_width
        for filter_size, stride in zip(filter_sizes, strides):
            current_size = int((current_size - filter_size) / stride) + 1
        flatten_shape = current_size * current_size * out_channels[-1]

        # flatten
        expected_output = np.full((self.batch_size, flatten_shape),
                                  filter_sum,
                                  dtype=np.float32)

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_sizes, in_channels, out_channels, '
                             'strides, pool_strides, pool_shapes', [
        ((1,), (3,), (32,), (1,), (1, 1), (1, 1)),  # noqa: E122
        ((3,), (3,), (32,), (1,), (2, 2), (1, 1)),
        ((3,), (3,), (32,), (1,), (1, 1), (2, 2)),
        ((3,), (3,), (32,), (1,), (2, 2), (2, 2)),
        ((3,), (3,), (32,), (2,), (1, 1), (2, 2)),
        ((3,), (3,), (32,), (2,), (2, 2), (2, 2)),
        ((1, 1), (3, 32), (32, 64), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (2, 2), (1, 1), (1, 1)),
    ])
    # yapf: enable
    def test_output_value_max_pooling(self, filter_sizes, in_channels,
                                      out_channels, strides, pool_strides,
                                      pool_shapes):
        model = CNNModelWithMaxPooling(
            filter_dims=filter_sizes,
            num_filters=out_channels,
            strides=strides,
            name='cnn_model',
            padding='VALID',
            pool_strides=pool_strides,
            pool_shapes=pool_shapes,
            hidden_w_init=tf.constant_initializer(1),
            hidden_nonlinearity=None)

        outputs = model.build(self._input_ph)
        output = self.sess.run(
            outputs, feed_dict={self._input_ph: self.obs_input})

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_size, in_channel in zip(filter_sizes, in_channels):
            filter_sum *= filter_size * filter_size * in_channel

        current_size = self.input_width
        for filter_size, stride in zip(filter_sizes, strides):
            current_size = int((current_size - filter_size) / stride) + 1
            current_size = int(
                (current_size - pool_shapes[0]) / pool_strides[0]) + 1

        flatten_shape = current_size * current_size * out_channels[-1]

        # flatten
        expected_output = np.full((self.batch_size, flatten_shape),
                                  filter_sum,
                                  dtype=np.float32)

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_sizes, in_channels, out_channels, '
                             'strides', [
        ((1, ), (3, ), (32, ), (1, )),  # noqa: E122
        ((3, ), (3, ), (32, ), (1, )),
        ((3, ), (3, ), (32, ), (2, )),
        ((1, 1), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filter_sizes, in_channels, out_channels,
                           strides):
        model = CNNModel(
            filter_dims=filter_sizes,
            num_filters=out_channels,
            strides=strides,
            name='cnn_model',
            padding='VALID',
            hidden_w_init=tf.constant_initializer(1),
            hidden_nonlinearity=None)
        outputs = model.build(self._input_ph)
        with tf.compat.v1.variable_scope('cnn_model/cnn/h0', reuse=True):
            bias = tf.compat.v1.get_variable('bias')
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(
            outputs, feed_dict={self._input_ph: self.obs_input})
        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)
            input_shape = self.obs_input.shape[1:]  # height, width, channel
            input_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(None, ) + input_shape, name='input')
            outputs = model_pickled.build(input_ph)
            output2 = sess.run(outputs, feed_dict={input_ph: self.obs_input})

            assert np.array_equal(output1, output2)

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
        self.obs_input = np.ones(
            (self.batch_size, self.input_width, self.input_height, 3))
        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self._input_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, ) + input_shape,
                                                  name='input')

    # yapf: disable
    @pytest.mark.parametrize('filters, in_channels, strides', [
        ((((1, 1), 32),), (3, ), (1, )),  # noqa: E122
        ((((3, 3), 32),), (3, ), (1, )),
        ((((3, 3), 32),), (3, ), (2, )),
        ((((1, 1), 32), ((1, 1), 64)), (3, 32), (1, 1)),
        ((((3, 3), 32), ((3, 3), 64)), (3, 32), (1, 1)),
        ((((3, 3), 32), ((3, 3), 64)), (3, 32), (2, 2)),
    ])
    # yapf: enable
    def test_output_value(self, filters, in_channels, strides):
        model = CNNModel(filters=filters,
                         strides=strides,
                         name='cnn_model',
                         padding='VALID',
                         hidden_w_init=tf.constant_initializer(1),
                         hidden_nonlinearity=None)

        outputs = model.build(self._input_ph)
        output = self.sess.run(outputs,
                               feed_dict={self._input_ph: self.obs_input})

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_iter, in_channel in zip(filters, in_channels):
            filter_sum *= filter_iter[0][0] * filter_iter[0][1] * in_channel

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[0][0]) / stride) + 1
            width_size = int((width_size - filter_iter[0][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][-1]

        # flatten
        expected_output = np.full((self.batch_size, flatten_shape),
                                  filter_sum,
                                  dtype=np.float32)

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize(
        'filters, in_channels, strides, pool_strides, pool_shapes',
        [
            ((((1, 1), 32), ), (3, ), (1, ), (1, 1), (1, 1)),  # noqa: E122
            ((((3, 3), 32), ), (3, ), (1, ), (2, 2), (1, 1)),
            ((((3, 3), 32), ), (3, ), (1, ), (1, 1), (2, 2)),
            ((((3, 3), 32), ), (3, ), (1, ), (2, 2), (2, 2)),
            ((((3, 3), 32), ), (3, ), (2, ), (1, 1), (2, 2)),
            ((((3, 3), 32), ), (3, ), (2, ), (2, 2), (2, 2)),
            ((((1, 1), 32), ((1, 1), 64)), (3, 32), (1, 1), (1, 1), (1, 1)),
            ((((3, 3), 32), ((3, 3), 64)), (3, 32), (1, 1), (1, 1), (1, 1)),
            ((((3, 3), 32), ((3, 3), 64)), (3, 32), (2, 2), (1, 1), (1, 1)),
        ])
    # yapf: enable
    def test_output_value_max_pooling(self, filters, in_channels, strides,
                                      pool_strides, pool_shapes):
        model = CNNModelWithMaxPooling(
            filters=filters,
            strides=strides,
            name='cnn_model',
            padding='VALID',
            pool_strides=pool_strides,
            pool_shapes=pool_shapes,
            hidden_w_init=tf.constant_initializer(1),
            hidden_nonlinearity=None)

        outputs = model.build(self._input_ph)
        output = self.sess.run(outputs,
                               feed_dict={self._input_ph: self.obs_input})

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_iter, in_channel in zip(filters, in_channels):
            filter_sum *= filter_iter[0][0] * filter_iter[0][1] * in_channel

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[0][0]) / stride) + 1
            height_size = int(
                (height_size - pool_shapes[0]) / pool_strides[0]) + 1
            width_size = int((width_size - filter_iter[0][1]) / stride) + 1
            width_size = int(
                (width_size - pool_shapes[1]) / pool_strides[1]) + 1

        flatten_shape = height_size * width_size * filters[-1][-1]

        # flatten
        expected_output = np.full((self.batch_size, flatten_shape),
                                  filter_sum,
                                  dtype=np.float32)

        assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filters, in_channels, strides', [
        ((((1, 1), 32),), (3, ), (1, )),  # noqa: E122
        ((((3, 3), 32),), (3, ), (1, )),
        ((((3, 3), 32),), (3, ), (2, )),
        ((((1, 1), 32), ((1, 1), 64)), (3, 32), (1, 1)),
        ((((3, 3), 32), ((3, 3), 64)), (3, 32), (1, 1)),
        ((((3, 3), 32), ((3, 3), 64)), (3, 32), (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filters, strides):
        model = CNNModel(filters=filters,
                         strides=strides,
                         name='cnn_model',
                         padding='VALID',
                         hidden_w_init=tf.constant_initializer(1),
                         hidden_nonlinearity=None)
        outputs = model.build(self._input_ph)
        with tf.compat.v1.variable_scope('cnn_model/cnn/h0', reuse=True):
            bias = tf.compat.v1.get_variable('bias')
        bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(outputs,
                                feed_dict={self._input_ph: self.obs_input})
        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)
            input_shape = self.obs_input.shape[1:]  # height, width, channel
            input_ph = tf.compat.v1.placeholder(tf.float32,
                                                shape=(None, ) + input_shape,
                                                name='input')
            outputs = model_pickled.build(input_ph)
            output2 = sess.run(outputs, feed_dict={input_ph: self.obs_input})

            assert np.array_equal(output1, output2)

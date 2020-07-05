import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models.cnn import cnn, cnn_with_max_pooling

from tests.fixtures import TfGraphTestCase
from tests.helpers import convolve, max_pooling


class TestCNN(TfGraphTestCase):

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
        self.hidden_nonlinearity = tf.nn.relu

    @pytest.mark.parametrize('filters, strides', [
        (((32, (1, 1)), ), (1, )),
        (((32, (3, 3)), ), (1, )),
        (((32, (2, 3)), ), (1, )),
        (((32, (3, 3)), ), (2, )),
        (((32, (2, 3)), ), (2, )),
        (((32, (1, 1)), (64, (1, 1))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (1, 1)),
        (((32, (2, 3)), (64, (3, 3))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (2, 2)),
        (((32, (2, 3)), (64, (3, 3))), (2, 2)),
    ])
    def test_output_shape_same(self, filters, strides):
        with tf.compat.v1.variable_scope('CNN'):
            self.cnn = cnn(input_var=self._input_ph,
                           filters=filters,
                           strides=strides,
                           name='cnn',
                           padding='SAME',
                           hidden_w_init=tf.constant_initializer(1),
                           hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        result = self.sess.run(self.cnn,
                               feed_dict={self._input_ph: self.obs_input})

        height_size = self.input_height
        width_size = self.input_width
        for stride in strides:
            height_size = int((height_size + stride - 1) / stride)
            width_size = int((width_size + stride - 1) / stride)
        flatten_shape = width_size * height_size * filters[-1][0]
        assert result.shape == (5, flatten_shape)

    @pytest.mark.parametrize('filters, strides', [
        (((32, (1, 1)), ), (1, )),
        (((32, (3, 3)), ), (1, )),
        (((32, (2, 3)), ), (1, )),
        (((32, (3, 3)), ), (2, )),
        (((32, (2, 3)), ), (2, )),
        (((32, (1, 1)), (64, (1, 1))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (1, 1)),
        (((32, (2, 3)), (64, (3, 3))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (2, 2)),
        (((32, (2, 3)), (64, (3, 3))), (2, 2)),
    ])
    def test_output_shape_valid(self, filters, strides):
        with tf.compat.v1.variable_scope('CNN'):
            self.cnn = cnn(input_var=self._input_ph,
                           filters=filters,
                           strides=strides,
                           name='cnn',
                           padding='VALID',
                           hidden_w_init=tf.constant_initializer(1),
                           hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        result = self.sess.run(self.cnn,
                               feed_dict={self._input_ph: self.obs_input})

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]
        assert result.shape == (self.batch_size, flatten_shape)

    @pytest.mark.parametrize('filters, in_channels, strides',
                             [(((32, (1, 1)), ), (3, ), (1, )),
                              (((32, (3, 3)), ), (3, ), (1, )),
                              (((32, (2, 3)), ), (3, ), (1, )),
                              (((32, (3, 3)), ), (3, ), (2, )),
                              (((32, (2, 3)), ), (3, ), (2, )),
                              (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1)),
                              (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1)),
                              (((32, (2, 3)), (64, (3, 3))), (3, 32), (1, 1)),
                              (((32, (3, 3)), (64, (3, 3))), (3, 32), (2, 2)),
                              (((32, (2, 3)), (64, (3, 3))), (3, 32), (2, 2))])
    def test_output_with_identity_filter(self, filters, in_channels, strides):
        with tf.compat.v1.variable_scope('CNN'):
            self.cnn = cnn(input_var=self._input_ph,
                           filters=filters,
                           strides=strides,
                           name='cnn1',
                           padding='VALID',
                           hidden_w_init=tf.constant_initializer(1),
                           hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        result = self.sess.run(self.cnn,
                               feed_dict={self._input_ph: self.obs_input})

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_iter, in_channel in zip(filters, in_channels):
            filter_sum *= filter_iter[1][0] * filter_iter[1][1] * in_channel

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
        flatten_shape = height_size * width_size * filters[-1][0]

        # flatten
        h_out = np.full((self.batch_size, flatten_shape),
                        filter_sum,
                        dtype=np.float32)
        np.testing.assert_array_equal(h_out, result)

    # yapf: disable
    @pytest.mark.parametrize('filters, in_channels, strides',
                             [(((32, (1, 1)), ), (3, ), (1, )),
                              (((32, (3, 3)), ), (3, ), (1, )),
                              (((32, (2, 3)), ), (3, ), (1, )),
                              (((32, (3, 3)), ), (3, ), (2, )),
                              (((32, (2, 3)), ), (3, ), (2, )),
                              (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1)),
                              (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1)),
                              (((32, (2, 3)), (64, (3, 3))), (3, 32), (1, 1)),
                              (((32, (3, 3)), (64, (3, 3))), (3, 32), (2, 2)),
                              (((32, (2, 3)), (64, (3, 3))), (3, 32), (2, 2))])
    # yapf: enable
    def test_output_with_random_filter(self, filters, in_channels, strides):
        # Build a cnn with random filter weights
        with tf.compat.v1.variable_scope('CNN'):
            self.cnn2 = cnn(input_var=self._input_ph,
                            filters=filters,
                            strides=strides,
                            name='cnn1',
                            padding='VALID',
                            hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        result = self.sess.run(self.cnn2,
                               feed_dict={self._input_ph: self.obs_input})

        two_layer = len(filters) == 2
        # get weight values
        with tf.compat.v1.variable_scope('CNN', reuse=True):
            h0_w = tf.compat.v1.get_variable('cnn1/h0/weight').eval()
            h0_b = tf.compat.v1.get_variable('cnn1/h0/bias').eval()
            if two_layer:
                h1_w = tf.compat.v1.get_variable('cnn1/h1/weight').eval()
                h1_b = tf.compat.v1.get_variable('cnn1/h1/bias').eval()
        filter_weights = (h0_w, h1_w) if two_layer else (h0_w, )
        filter_bias = (h0_b, h1_b) if two_layer else (h0_b, )

        # convolution according to TensorFlow's approach
        input_val = convolve(_input=self.obs_input,
                             filter_weights=filter_weights,
                             filter_bias=filter_bias,
                             strides=strides,
                             filters=filters,
                             in_channels=in_channels,
                             hidden_nonlinearity=self.hidden_nonlinearity)

        # flatten
        dense_out = input_val.reshape((self.batch_size, -1)).astype(np.float32)
        np.testing.assert_array_almost_equal(dense_out, result)

    # yapf: disable
    @pytest.mark.parametrize(
        'filters, in_channels, strides, pool_shape, pool_stride', [
            (((32, (1, 1)), ), (3, ), (1, ), 1, 1),
            (((32, (3, 3)), ), (3, ), (1, ), 1, 1),
            (((32, (2, 3)), ), (3, ), (1, ), 1, 1),
            (((32, (3, 3)), ), (3, ), (2, ), 2, 2),
            (((32, (2, 3)), ), (3, ), (2, ), 2, 2),
            (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1), 1, 1),
            (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1), 1, 1),
            (((32, (2, 3)), (64, (3, 3))), (3, 32), (1, 1), 1, 1)
        ])
    # yapf: enable
    def test_output_with_max_pooling(self, filters, in_channels, strides,
                                     pool_shape, pool_stride):
        # Build a cnn with random filter weights
        with tf.compat.v1.variable_scope('CNN'):
            self.cnn2 = cnn_with_max_pooling(
                input_var=self._input_ph,
                filters=filters,
                strides=strides,
                name='cnn1',
                pool_shapes=(pool_shape, pool_shape),
                pool_strides=(pool_stride, pool_stride),
                padding='VALID',
                hidden_w_init=tf.constant_initializer(1),
                hidden_nonlinearity=self.hidden_nonlinearity)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        result = self.sess.run(self.cnn2,
                               feed_dict={self._input_ph: self.obs_input})

        two_layer = len(filters) == 2
        # get weight values
        with tf.compat.v1.variable_scope('CNN', reuse=True):
            h0_w = tf.compat.v1.get_variable('cnn1/h0/weight').eval()
            h0_b = tf.compat.v1.get_variable('cnn1/h0/bias').eval()
            if two_layer:
                h1_w = tf.compat.v1.get_variable('cnn1/h1/weight').eval()
                h1_b = tf.compat.v1.get_variable('cnn1/h1/bias').eval()

        filter_weights = (h0_w, h1_w) if two_layer else (h0_w, )
        filter_bias = (h0_b, h1_b) if two_layer else (h0_b, )

        input_val = self.obs_input

        # convolution according to TensorFlow's approach
        # and perform max pooling on each layer
        for filter_iter, filter_weight, _filter_bias, in_channel in zip(
                filters, filter_weights, filter_bias, in_channels):
            input_val = convolve(_input=input_val,
                                 filter_weights=(filter_weight, ),
                                 filter_bias=(_filter_bias, ),
                                 strides=strides,
                                 filters=(filter_iter, ),
                                 in_channels=(in_channel, ),
                                 hidden_nonlinearity=self.hidden_nonlinearity)

            # max pooling
            input_val = max_pooling(_input=input_val,
                                    pool_shape=pool_shape,
                                    pool_stride=pool_stride)

        # flatten
        dense_out = input_val.reshape((self.batch_size, -1)).astype(np.float32)
        np.testing.assert_array_equal(dense_out, result)

    def test_invalid_padding(self):
        with pytest.raises(ValueError):
            with tf.compat.v1.variable_scope('CNN'):
                self.cnn = cnn(input_var=self._input_ph,
                               filters=((32, (3, 3)), ),
                               strides=(1, ),
                               name='cnn',
                               padding='UNKNOWN')

    def test_invalid_padding_max_pooling(self):
        with pytest.raises(ValueError):
            with tf.compat.v1.variable_scope('CNN'):
                self.cnn = cnn_with_max_pooling(input_var=self._input_ph,
                                                filters=((32, (3, 3)), ),
                                                strides=(1, ),
                                                name='cnn',
                                                pool_shapes=(1, 1),
                                                pool_strides=(1, 1),
                                                padding='UNKNOWN')

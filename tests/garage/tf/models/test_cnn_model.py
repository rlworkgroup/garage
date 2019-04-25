import pickle

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.models import CNNModel
from tests.fixtures import TfGraphTestCase


class TestCNNModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 5
        self.input_width = 10
        self.input_height = 10
        self.obs_input = np.ones((self.batch_size, self.input_width,
                                  self.input_height, 3))
        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self._input_ph = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape, name="input")

    @params(
        ((1, ), (3, ), (32, ), (1, )),
        ((3, ), (3, ), (32, ), (1, )),
        ((3, ), (3, ), (32, ), (2, )),
        ((1, 1), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (2, 2)),
    )
    def test_output_value(self, filter_sizes, in_channels, out_channels,
                          strides):
        model = CNNModel(
            filter_dims=filter_sizes,
            num_filters=out_channels,
            strides=strides,
            name="cnn_model",
            padding="VALID",
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

    @params(
        ((1, ), (3, ), (32, ), (1, )),
        ((3, ), (3, ), (32, ), (1, )),
        ((3, ), (3, ), (32, ), (2, )),
        ((1, 1), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (1, 1)),
        ((3, 3), (3, 32), (32, 64), (2, 2)),
    )
    def test_is_pickleable(self, filter_sizes, in_channels, out_channels,
                           strides):
        model = CNNModel(
            filter_dims=filter_sizes,
            num_filters=out_channels,
            strides=strides,
            name="cnn_model",
            padding="VALID",
            hidden_w_init=tf.constant_initializer(1),
            hidden_nonlinearity=None)
        outputs = model.build(self._input_ph)
        with tf.variable_scope('cnn_model/cnn/h0', reuse=True):
            bias = tf.get_variable('bias')
        self.sess.run(tf.assign(bias, tf.ones_like(bias)))

        output1 = self.sess.run(
            outputs, feed_dict={self._input_ph: self.obs_input})
        h = pickle.dumps(model)
        with tf.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)
            input_shape = self.obs_input.shape[1:]  # height, width, channel
            input_ph = tf.placeholder(
                tf.float32, shape=(None, ) + input_shape, name="input")
            outputs = model_pickled.build(input_ph)
            output2 = sess.run(outputs, feed_dict={input_ph: self.obs_input})

            assert np.array_equal(output1, output2)

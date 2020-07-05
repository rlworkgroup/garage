import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models import CNNMLPMergeModel

from tests.fixtures import TfGraphTestCase


class TestCNNMLPMergeModel(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.batch_size = 1
        self.input_width = 10
        self.input_height = 10

        self.obs_input = np.ones(
            (self.batch_size, self.input_width, self.input_height, 3))

        # skip batch size
        self.input_shape = (self.input_width, self.input_height, 3)

        self.obs_ph = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, ) +
                                               self.input_shape,
                                               name='input')
        self.action_input = np.ones((self.batch_size, 3))

        self.action_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, ) +
                                                  self.action_input.shape[1:],
                                                  name='input')
        self.action_input = np.ones((self.batch_size, 3))
        self.hidden_nonlinearity = tf.nn.relu

    # yapf: disable
    @pytest.mark.parametrize('filters, in_channels, strides, hidden_sizes', [
        (((32, (1, 1)), ), (3, ), (1, ), (1, )),  # noqa: E122
        (((32, (3, 3)), ), (3, ), (1, ), (2, )),
        (((32, (3, 3)), ), (3, ), (2, ), (3, )),
        (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1), (1, )),
        (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1), (2, )),
        (((32, (3, 3)), (64, (3, 3))), (3, 32), (2, 2), (3, )),
    ])
    # yapf: enable
    def test_output_value(self, filters, in_channels, strides, hidden_sizes):
        model = CNNMLPMergeModel(filters=filters,
                                 strides=strides,
                                 hidden_sizes=hidden_sizes,
                                 action_merge_layer=1,
                                 name='cnn_mlp_merge_model1',
                                 padding='VALID',
                                 cnn_hidden_w_init=tf.constant_initializer(1),
                                 hidden_nonlinearity=self.hidden_nonlinearity)

        model_out = model.build(self.obs_ph, self.action_ph).outputs
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
        cnn_output = np.full((self.batch_size, flatten_shape),
                             filter_sum,
                             dtype=np.float32)

        with tf.compat.v1.variable_scope('cnn_mlp_merge_model1/MLPMergeModel',
                                         reuse=True):
            h0_w = tf.compat.v1.get_variable('mlp_concat/hidden_0/kernel')
            h0_b = tf.compat.v1.get_variable('mlp_concat/hidden_0/bias')
            out_w = tf.compat.v1.get_variable('mlp_concat/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp_concat/output/bias')

        mlp_output = self.sess.run(model_out,
                                   feed_dict={
                                       self.obs_ph: self.obs_input,
                                       self.action_ph: self.action_input
                                   })

        # First layer
        h0_in = tf.matmul(cnn_output, h0_w) + h0_b
        h0_out = self.hidden_nonlinearity(h0_in)

        # output
        h1_in = tf.matmul(tf.concat([h0_out, self.action_input], 1),
                          out_w) + out_b

        # eval output
        out = self.sess.run(h1_in,
                            feed_dict={
                                self.obs_ph: self.obs_input,
                                self.action_ph: self.action_input
                            })

        np.testing.assert_array_equal(out, mlp_output)

    @pytest.mark.parametrize(
        'filters, in_channels, strides, pool_strides, pool_shapes',
        [
            (((32, (1, 1)), ), (3, ), (1, ), (1, 1), (1, 1)),  # noqa: E122
            (((32, (3, 3)), ), (3, ), (1, ), (2, 2), (1, 1)),
            (((32, (3, 3)), ), (3, ), (1, ), (1, 1), (2, 2)),
            (((32, (3, 3)), ), (3, ), (1, ), (2, 2), (2, 2)),
            (((32, (3, 3)), ), (3, ), (2, ), (1, 1), (2, 2)),
            (((32, (3, 3)), ), (3, ), (2, ), (2, 2), (2, 2)),
            (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1), (1, 1), (1, 1)),
            (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1), (1, 1), (1, 1)),
            (((32, (3, 3)), (64, (3, 3))), (3, 32), (2, 2), (1, 1), (1, 1)),
        ])
    def test_output_value_max_pooling(self, filters, in_channels, strides,
                                      pool_strides, pool_shapes):
        model = CNNMLPMergeModel(filters=filters,
                                 strides=strides,
                                 name='cnn_mlp_merge_model2',
                                 padding='VALID',
                                 max_pooling=True,
                                 action_merge_layer=1,
                                 pool_strides=pool_strides,
                                 pool_shapes=pool_shapes,
                                 cnn_hidden_w_init=tf.constant_initializer(1),
                                 hidden_nonlinearity=self.hidden_nonlinearity)

        model_out = model.build(self.obs_ph, self.action_ph).outputs

        filter_sum = 1
        # filter value after 3 layers of conv
        for filter_iter, in_channel in zip(filters, in_channels):
            filter_sum *= filter_iter[1][0] * filter_iter[1][1] * in_channel

        height_size = self.input_height
        width_size = self.input_width
        for filter_iter, stride in zip(filters, strides):
            height_size = int((height_size - filter_iter[1][0]) / stride) + 1
            height_size = int(
                (height_size - pool_shapes[0]) / pool_strides[0]) + 1
            width_size = int((width_size - filter_iter[1][1]) / stride) + 1
            width_size = int(
                (width_size - pool_shapes[1]) / pool_strides[1]) + 1

        flatten_shape = height_size * width_size * filters[-1][0]

        # flatten
        cnn_output = np.full((self.batch_size, flatten_shape),
                             filter_sum,
                             dtype=np.float32)

        # feed cnn output to MLPMergeModel
        with tf.compat.v1.variable_scope('cnn_mlp_merge_model2/MLPMergeModel',
                                         reuse=True):
            h0_w = tf.compat.v1.get_variable('mlp_concat/hidden_0/kernel')
            h0_b = tf.compat.v1.get_variable('mlp_concat/hidden_0/bias')
            out_w = tf.compat.v1.get_variable('mlp_concat/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp_concat/output/bias')

        mlp_output = self.sess.run(model_out,
                                   feed_dict={
                                       self.obs_ph: self.obs_input,
                                       self.action_ph: self.action_input
                                   })

        # First layer
        h0_in = tf.matmul(cnn_output, h0_w) + h0_b
        h0_out = self.hidden_nonlinearity(h0_in)

        # output
        h1_in = tf.matmul(tf.concat([h0_out, self.action_input], 1),
                          out_w) + out_b

        # eval output
        out = self.sess.run(h1_in,
                            feed_dict={
                                self.obs_ph: self.obs_input,
                                self.action_ph: self.action_input
                            })

        np.testing.assert_array_equal(out, mlp_output)

    # yapf: disable
    @pytest.mark.parametrize('filters, strides', [
        (((32, (1, 1)), ), (1, )),  # noqa: E122
        (((32, (3, 3)), ), (1, )),
        (((32, (3, 3)), ), (2, )),
        (((32, (1, 1)), (64, (1, 1))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filters, strides):
        model = CNNMLPMergeModel(filters=filters,
                                 strides=strides,
                                 name='cnn_mlp_merge_model',
                                 padding='VALID',
                                 cnn_hidden_w_init=tf.constant_initializer(1),
                                 hidden_nonlinearity=None)
        outputs = model.build(self.obs_ph).outputs

        with tf.compat.v1.variable_scope(
                'cnn_mlp_merge_model/MLPMergeModel/mlp_concat', reuse=True):
            bias = tf.compat.v1.get_variable('output/bias')
            bias.load(tf.ones_like(bias).eval())

        output1 = self.sess.run(outputs,
                                feed_dict={self.obs_ph: self.obs_input})
        h = pickle.dumps(model)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            model_pickled = pickle.loads(h)
            input_ph = tf.compat.v1.placeholder(tf.float32,
                                                shape=(None, ) +
                                                self.input_shape,
                                                name='input')
            outputs = model_pickled.build(input_ph).outputs
            output2 = sess.run(outputs, feed_dict={input_ph: self.obs_input})

            assert np.array_equal(output1, output2)

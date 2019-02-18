import pickle

import numpy as np
import tensorflow as tf

from garage.tf.core.mlp import mlp
from garage.tf.models.base import TfModel
from tests.fixtures import TfGraphTestCase


class SimpleModel(TfModel):
    def __init__(self, output_dim=2, hidden_sizes=(4, 4), name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes

    def network_output_spec(self):
        return ['state', 'action']

    def _build(self, obs_input):
        state = mlp(obs_input, self._output_dim, self._hidden_sizes, 'state')
        action = mlp(obs_input, self._output_dim, self._hidden_sizes, 'action')
        return state, action


class ComplicatedModel(TfModel):
    def __init__(self, output_dim=2, name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._simple_model_1 = SimpleModel(output_dim=4)
        self._simple_model_2 = SimpleModel(
            output_dim=output_dim, name='simple_model_2')

    def network_output_spec(self):
        return ['action']

    def _build(self, obs_input):
        h1, _ = self._simple_model_1.build(obs_input)
        return self._simple_model_2.build(h1)[1]


class TestModel(TfGraphTestCase):
    def test_model_creation(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        outputs = model.build(input_var)
        data = np.ones((3, 5))
        out, model_out = self.sess.run(
            [outputs, model.networks['default'].outputs],
            feed_dict={model.networks['default'].input: data})
        assert np.array_equal(out, model_out)

    def test_model_creation_with_custom_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        outputs = model.build(input_var, name='network_2')
        data = np.ones((3, 5))
        result, result2 = self.sess.run(
            [outputs, model.networks['network_2'].outputs],
            feed_dict={model.networks['network_2'].input: data})
        assert np.array_equal(result, result2)

    def test_same_model_with_no_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var)
        with self.assertRaises(ValueError):
            model.build(another_input_var)

        model2 = SimpleModel(output_dim=2)
        with self.assertRaises(ValueError):
            model2.build(another_input_var)

    def test_model_with_different_name(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        outputs_1 = model.build(input_var)
        outputs_2 = model.build(another_input_var, name='network_2')
        data = np.ones((3, 5))
        results_1, results_2 = self.sess.run([outputs_1, outputs_2],
                                             feed_dict={
                                                 input_var: data,
                                                 another_input_var: data
                                             })  # noqa: E126
        assert np.array_equal(results_1, results_2)

    def test_model_with_different_name_in_different_order(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        outputs_1 = model.build(input_var, name='network_1')
        outputs_2 = model.build(another_input_var)
        data = np.ones((3, 5))
        results_1, results_2 = self.sess.run([outputs_1, outputs_2],
                                             feed_dict={
                                                 input_var: data,
                                                 another_input_var: data
                                             })  # noqa: E126
        assert np.array_equal(results_1, results_2)

    def test_model_in_model(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        model = ComplicatedModel(output_dim=2)
        outputs = model.build(input_var)
        data = np.ones((3, 5))
        out, model_out = self.sess.run(
            [outputs, model.networks['default'].outputs],
            feed_dict={model.networks['default'].input: data})
        assert np.array_equal(out, model_out)

    def test_model_pickle(self):
        data = np.ones((3, 5))
        model = SimpleModel(output_dim=2)

        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model.build(input_var)

            results = sess.run(
                model.networks['default'].outputs,
                feed_dict={model.networks['default'].input: data})
            model_data = pickle.dumps(model)

        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            outputs = model_pickled.build(input_var)

            results2 = sess.run(outputs, feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_model_in_model_pickle_same_parameters(self):
        data = np.ones((3, 5))
        model = ComplicatedModel(output_dim=2)

        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            outputs = model.build(input_var)

            results = sess.run(
                outputs, feed_dict={model.networks['default'].input: data})
            model_data = pickle.dumps(model)

        with tf.Session(graph=tf.Graph()) as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            model_pickled.build(input_var)

            results2 = sess.run(
                model_pickled.networks['default'].outputs,
                feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_model_pickle_same_parameters(self):
        model = SimpleModel(output_dim=2)

        with tf.Session(graph=tf.Graph()):
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            model.build(state)

            model.parameters = {
                k: np.zeros_like(v)
                for k, v in model.parameters.items()
            }
            all_one = {k: np.ones_like(v) for k, v in model.parameters.items()}
            model.parameters = all_one
            h_data = pickle.dumps(model)

        with tf.Session(graph=tf.Graph()):
            model_pickled = pickle.loads(h_data)
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            model_pickled.build(state)

            np.testing.assert_equal(all_one, model_pickled.parameters)

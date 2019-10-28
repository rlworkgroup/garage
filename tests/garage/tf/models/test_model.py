import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.mlp import mlp
from tests.fixtures import TfGraphTestCase


class SimpleModel(Model):

    def __init__(self, output_dim=2, hidden_sizes=(4, 4), name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes

    def network_output_spec(self):
        return ['state', 'action']

    def _build(self, obs_input, name=None):
        state = mlp(obs_input, self._output_dim, self._hidden_sizes, 'state')
        action = mlp(obs_input, self._output_dim, self._hidden_sizes, 'action')
        return state, action


# This model doesn't implement network_output_spec
class SimpleModel2(Model):

    def __init__(self, output_dim=2, hidden_sizes=(4, 4), name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes

    def _build(self, obs_input, name=None):
        action = mlp(obs_input, self._output_dim, self._hidden_sizes, 'state')
        return action


class ComplicatedModel(Model):

    def __init__(self, output_dim=2, name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._simple_model_1 = SimpleModel(output_dim=4)
        self._simple_model_2 = SimpleModel2(output_dim=output_dim,
                                            name='simple_model_2')

    def network_output_spec(self):
        return ['action']

    def _build(self, obs_input, name=None):
        h1, _ = self._simple_model_1.build(obs_input)
        return self._simple_model_2.build(h1)


# This model takes another model as constructor argument
class ComplicatedModel2(Model):

    def __init__(self, parent_model, output_dim=2, name=None):
        super().__init__(name)
        self._output_dim = output_dim
        self._parent_model = parent_model
        self._output_model = SimpleModel2(output_dim=output_dim)

    def network_output_spec(self):
        return ['action']

    def _build(self, obs_input, name=None):
        h1, _ = self._parent_model.build(obs_input)
        return self._output_model.build(h1)


class TestModel(TfGraphTestCase):

    def test_model_creation(self):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2)
        outputs = model.build(input_var)
        data = np.ones((3, 5))
        out, model_out = self.sess.run(
            [outputs, model.networks['default'].outputs],
            feed_dict={model.networks['default'].input: data})
        assert np.array_equal(out, model_out)
        assert model.name == type(model).__name__

    def test_model_creation_with_custom_name(self):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        model = SimpleModel(output_dim=2, name='MySimpleModel')
        outputs = model.build(input_var, name='network_2')
        data = np.ones((3, 5))
        result, result2 = self.sess.run(
            [outputs, model.networks['network_2'].outputs],
            feed_dict={model.networks['network_2'].input: data})
        assert np.array_equal(result, result2)
        assert model.name == 'MySimpleModel'

    def test_same_model_with_no_name(self):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.compat.v1.placeholder(tf.float32,
                                                     shape=(None, 5))
        model = SimpleModel(output_dim=2)
        model.build(input_var)
        with pytest.raises(ValueError):
            model.build(another_input_var)

        model2 = SimpleModel(output_dim=2)
        with pytest.raises(ValueError):
            model2.build(another_input_var)

    def test_model_with_different_name(self):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.compat.v1.placeholder(tf.float32,
                                                     shape=(None, 5))
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
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        another_input_var = tf.compat.v1.placeholder(tf.float32,
                                                     shape=(None, 5))
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
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        model = ComplicatedModel(output_dim=2)
        outputs = model.build(input_var)
        data = np.ones((3, 5))
        out, model_out = self.sess.run(
            [outputs, model.networks['default'].outputs],
            feed_dict={model.networks['default'].input: data})

        assert np.array_equal(out, model_out)

    def test_model_as_constructor_argument(self):
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
        parent_model = SimpleModel(output_dim=4)
        model = ComplicatedModel2(parent_model=parent_model, output_dim=2)
        outputs = model.build(input_var)
        data = np.ones((3, 5))
        out, model_out = self.sess.run(
            [outputs, model.networks['default'].outputs],
            feed_dict={model.networks['default'].input: data})

        assert np.array_equal(out, model_out)

    def test_model_is_pickleable(self):
        data = np.ones((3, 5))
        model = SimpleModel(output_dim=2)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model.build(input_var)

            # assign bias to all one
            with tf.compat.v1.variable_scope('SimpleModel/state', reuse=True):
                bias = tf.compat.v1.get_variable('hidden_0/bias')
            bias.load(tf.ones_like(bias).eval())

            results = sess.run(
                model.networks['default'].outputs,
                feed_dict={model.networks['default'].input: data})
            model_data = pickle.dumps(model)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            outputs = model_pickled.build(input_var)

            results2 = sess.run(outputs, feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_model_pickle_without_building(self):
        model = SimpleModel(output_dim=2)
        model_data = pickle.dumps(model)
        model_pickled = pickle.loads(model_data)

        assert np.array_equal(model.name, model_pickled.name)

    def test_complicated_model_is_pickleable(self):
        data = np.ones((3, 5))

        model = ComplicatedModel(output_dim=2)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            outputs = model.build(input_var)

            # assign bias to all one
            with tf.compat.v1.variable_scope(
                    'ComplicatedModel/SimpleModel/state', reuse=True):
                bias = tf.compat.v1.get_variable('hidden_0/bias')
            bias.load(tf.ones_like(bias).eval())

            results = sess.run(
                outputs, feed_dict={model.networks['default'].input: data})
            model_data = pickle.dumps(model)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            model_pickled.build(input_var)

            results2 = sess.run(model_pickled.networks['default'].outputs,
                                feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_complicated_model2_is_pickleable(self):
        data = np.ones((3, 5))

        parent_model = SimpleModel(output_dim=4)
        model = ComplicatedModel2(parent_model=parent_model, output_dim=2)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            outputs = model.build(input_var)

            # assign bias to all one
            with tf.compat.v1.variable_scope(
                    'ComplicatedModel2/SimpleModel/state', reuse=True):
                bias = tf.compat.v1.get_variable('hidden_0/bias')
            bias.load(tf.ones_like(bias).eval())

            results = sess.run(
                outputs, feed_dict={model.networks['default'].input: data})
            model_data = pickle.dumps(model)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            input_var = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            model_pickled.build(input_var)

            results2 = sess.run(model_pickled.networks['default'].outputs,
                                feed_dict={input_var: data})

        assert np.array_equal(results, results2)

    def test_simple_model_is_pickleable_with_same_parameters(self):
        model = SimpleModel(output_dim=2)

        with tf.compat.v1.Session(graph=tf.Graph()):
            state = tf.compat.v1.placeholder(shape=[None, 10, 5],
                                             dtype=tf.float32)
            model.build(state)

            model.parameters = {
                k: np.zeros_like(v)
                for k, v in model.parameters.items()
            }
            all_one = {k: np.ones_like(v) for k, v in model.parameters.items()}
            model.parameters = all_one
            h_data = pickle.dumps(model)

        with tf.compat.v1.Session(graph=tf.Graph()):
            model_pickled = pickle.loads(h_data)
            state = tf.compat.v1.placeholder(shape=[None, 10, 5],
                                             dtype=tf.float32)
            model_pickled.build(state)

            np.testing.assert_equal(all_one, model_pickled.parameters)

    def test_simple_model_is_pickleable_with_missing_parameters(self):
        model = SimpleModel(output_dim=2)

        with tf.compat.v1.Session(graph=tf.Graph()):
            state = tf.compat.v1.placeholder(shape=[None, 10, 5],
                                             dtype=tf.float32)
            model.build(state)

            model.parameters = {
                k: np.zeros_like(v)
                for k, v in model.parameters.items()
            }
            all_one = {k: np.ones_like(v) for k, v in model.parameters.items()}
            model.parameters = all_one
            h_data = pickle.dumps(model)

        with tf.compat.v1.Session(graph=tf.Graph()):
            model_pickled = pickle.loads(h_data)
            state = tf.compat.v1.placeholder(shape=[None, 10, 5],
                                             dtype=tf.float32)
            # remove one of the parameters
            del model_pickled._default_parameters[
                'SimpleModel/state/hidden_0/kernel:0']
            with pytest.warns(UserWarning):
                model_pickled.build(state)

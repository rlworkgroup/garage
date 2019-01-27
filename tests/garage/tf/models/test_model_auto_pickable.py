import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from garage.tf.core.parameter_layer import ParameterLayer
from garage.tf.models import GaussianMLPModel
from garage.tf.models import MLPModel
from tests.fixtures import TfGraphTestCase


class TestKerasModel(TfGraphTestCase):
    def test_parameter_layer_pickling_from_json(self):
        input_var = Input(shape=(5, ))

        parameter = ParameterLayer(length=2)(input_var)
        model = Model(inputs=input_var, outputs=parameter)

        self.sess.run(tf.global_variables_initializer())

        fresh_model = tf.keras.models.model_from_json(
            model.to_json(), custom_objects={'ParameterLayer': ParameterLayer})
        fresh_model.set_weights(model.get_weights())

        model_output = self.sess.run(
            model.output, feed_dict={model.input: np.random.random((2, 5))})
        fresh_model_output = self.sess.run(
            fresh_model.output,
            feed_dict={fresh_model.input: np.random.random((2, 5))})

        assert np.array_equal(model_output, fresh_model_output)

    # Type Error: Not JSON Serializable
    def test_pickling_from_json_with_tensor_argument(self):
        input_var = Input(shape=(5, ))
        initializer = tf.keras.initializers.constant(2.0)

        parameter = ParameterLayer(
            length=2, initializer=initializer)(input_var)
        model = Model(inputs=input_var, outputs=parameter)

        self.sess.run(tf.global_variables_initializer())

        with self.assertRaises(TypeError):
            tf.keras.models.model_from_json(
                model.to_json(),
                custom_objects={"ParameterLayer": ParameterLayer})

    def test_parameter_layer_pickling_from_config(self):
        input_var = Input(shape=(5, ))

        parameter = ParameterLayer(length=2)(input_var)
        model = Model(inputs=input_var, outputs=parameter)

        self.sess.run(tf.global_variables_initializer())

        fresh_model = tf.keras.models.Model.from_config(
            model.get_config(),
            custom_objects={"ParameterLayer": ParameterLayer})
        fresh_model.set_weights(model.get_weights())

        model_output = self.sess.run(
            model.output, feed_dict={model.input: np.random.random((2, 5))})
        model_pickled_output = self.sess.run(
            fresh_model.output,
            feed_dict={fresh_model.input: np.random.random((2, 5))})

        assert np.array_equal(model_output, model_pickled_output)

    def test_autopickable_mlp_pickling(self):
        mlp = MLPModel(input_dim=5, output_dim=2, hidden_sizes=(4, 4))

        self.sess.run(tf.global_variables_initializer())

        fresh_mlp = pickle.loads(pickle.dumps(mlp))

        data = np.random.random((2, 5))

        model_output = self.sess.run(mlp.output, feed_dict={mlp.input: data})
        model_pickled_output = self.sess.run(
            fresh_mlp.output, feed_dict={fresh_mlp.input: data})

        assert np.array_equal(model_output, model_pickled_output)

    def test_autopickable_gaussian_mlp_pickling(self):
        model = GaussianMLPModel(
            input_dim=5, output_dim=2, hidden_sizes=(4, 4), init_std=2.0)

        self.sess.run(tf.global_variables_initializer())

        data = np.random.random((3, 5))

        result_from_model = []
        result_from_model.append(
            self.sess.run(model.outputs, feed_dict={model.input: data}))

        model_pickled = pickle.loads(pickle.dumps(model))

        result_from_pickled_model = []
        result_from_pickled_model.append(
            self.sess.run(
                model_pickled.outputs, feed_dict={model_pickled.input: data}))

        assert np.array_equal(result_from_model, result_from_pickled_model)

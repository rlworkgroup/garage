"""MLP Layer based on tf.keras.layer."""
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tests.fixtures import TfGraphTestCase

# flake8: noqa
# pylint: noqa


def mlp(input_var,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity='relu',
        hidden_w_init='glorot_uniform',
        hidden_b_init='zeros',
        output_nonlinearity=None,
        output_w_init='glorot_uniform',
        output_b_init='zero',
        batch_normalization=False,
        **kwargs):
    x = input_var
    if isinstance(hidden_sizes, int):
        x = Dense(
            units=hidden_sizes,
            activation=hidden_nonlinearity,
            kernel_initializer=hidden_w_init,
            bias_initializer=hidden_b_init)(x)
        if batch_normalization:
            x = BatchNormalization(
                hidden_sizes, activation=hidden_nonlinearity)(x)
    else:
        x = Dense(
            units=hidden_sizes[0],
            activation=hidden_nonlinearity,
            kernel_initializer=hidden_w_init,
            bias_initializer=hidden_b_init)(x)
        if batch_normalization:
            x = BatchNormalization(
                hidden_sizes[0], activation=hidden_nonlinearity)(x)
        for hidden_size in hidden_sizes[1:]:
            x = Dense(
                units=hidden_size,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init)(x)
            if batch_normalization:
                x = BatchNormalization(
                    hidden_size, activation=hidden_nonlinearity)(x)
    x = Dense(
        units=output_dim,
        activation=output_nonlinearity,
        kernel_initializer=output_w_init,
        bias_initializer=output_b_init)(x)
    return Model(inputs=input_var, outputs=x)


# option 1
def GaussianMLPModel(input_var, *args, **kwargs):
    mean = mlp(input_var, *args, **kwargs)
    std = mlp(input_var, *args, **kwargs)
    dist = Add()([mean.output, std.output])
    return Model(inputs=input_var, outputs=[mean.output, std.output, dist])


# option 2
class GaussianMLPModelC:
    def __getstate__(self):
        return {'model': self.model.to_json()}

    def __setstate__(self, d):
        model = tf.keras.models.model_from_json(self.__dict__['model'])
        self.__dict__.update(model.__dict__)

    def __init__(self, input_var, *args, **kwargs):
        self.model = self.build_model(input_var, *args, **kwargs)
        self._mean = self.model.outputs[0]
        self._std = self.model.outputs[1]
        self._dist = self.model.outputs[2]
        self._input = input_var

    def dist(self):
        return self._dist

    def input(self):
        return self._input

    def mean(self):
        return self._mean

    def build_model(self, input_var, *args, **kwargs):
        mean = mlp(input_var, *args, **kwargs)
        std = mlp(input_var, *args, **kwargs)
        dist = Add()([mean.output, std.output])
        return Model(inputs=input_var, outputs=[mean.output, std.output, dist])


class TestKerasModel(TfGraphTestCase):
    def test_gaussian_mlp(self):
        input_var = Input(shape=(5, ))
        model = GaussianMLPModel(
            input_var=input_var, output_dim=2, hidden_sizes=(4, 4))

        modelC = GaussianMLPModelC(
            input_var=input_var, output_dim=2, hidden_sizes=(4, 4))

        pickle.dumps(model)

    def test_mlp(self):
        input_var = Input(shape=(5, ))
        model = mlp(input_var, 2, (4))
        self.sess.run(tf.global_variables_initializer())
        data = np.random.random((2, 5))
        y = self.sess.run(model.output, feed_dict={model.input: data})

        print("\n### Keras custom model inputs: {}\n".format(model.inputs))
        print("\n### Keras custom model outputs: {}\n".format(model.outputs))
        print("\n### Keras custom model : {}\n".format(model.get_config()))

        print("\nJSON : {}\n".format(model.to_json()))
        fresh_model = tf.keras.models.model_from_json(model.to_json())
        print("\nRestore from JSON : {}\n".format(fresh_model.get_config()))

        print("\nYAML : {}\n".format(model.to_yaml()))
        fresh_model = tf.keras.models.model_from_yaml(model.to_yaml())
        print("\nRestore from YAML: {}\n".format(fresh_model.get_config()))
        # print("\nJSON : {}\n".format(model.to_json()))
        # fresh_model = tf.keras.models.model_from_json(
        #     model.to_json(), custom_objects={'MLPLayer': MLPLayer})
        # print("\nRestore from JSON : {}\n".format(fresh_model.get_config()))

        # print("\nYAML : {}\n".format(model.to_yaml()))
        # fresh_model = tf.keras.models.model_from_yaml(
        #     model.to_yaml(), custom_objects={'MLPLayer': MLPLayer})
        # print("\nRestore from YAML: {}\n".format(fresh_model.get_config()))

        print("\n### Saving/loading weights...")
        weights = model.get_weights()
        weights[2] = np.ones_like(weights[2])
        model.set_weights(weights)
        print("\nweights being saved: {}".format(model.get_weights()))
        model.save_weights('./test_keras.weights')

        print("\nweights before load: {}".format(fresh_model.get_weights()))
        fresh_model.load_weights('./test_keras.weights')
        print("\nweights after load: {}".format(fresh_model.get_weights()))

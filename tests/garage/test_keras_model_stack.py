"""MLP Layer based on tf.keras.layer."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import tensor_shape

# flake8: noqa


class MLPLayer(KerasLayer):
    def __init__(self,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init='uniform',
                 hidden_b_init='zero',
                 output_nonlinearity=None,
                 output_w_init='uniform',
                 output_b_init='zero',
                 layer_normalization=False,
                 **kwargs):
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_w_init = hidden_w_init
        self.hidden_b_init = hidden_b_init
        self.output_nonlinearity = output_nonlinearity
        self.output_w_init = output_w_init
        self.output_b_init = output_b_init
        self.layer_normalization = layer_normalization
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.kernels = []
        self.bias = []
        if isinstance(self.hidden_sizes, int):
            kernel = self.add_weight(
                name='kernel_0',
                shape=[input_shape[-1].value, self.output_dim],
                initializer=self.output_w_init,
                dtype=tf.float32)
            bias = self.add_weight(
                name='bias_0',
                shape=(self.output_dim),
                initializer=self.output_b_init)
            self.kernels.append(kernel)
            self.bias.append(bias)
        else:
            _hidden_size = self.hidden_sizes[0]
            kernel = self.add_weight(
                name='kernel_0',
                shape=[input_shape[-1].value, _hidden_size],
                initializer=self.hidden_w_init,
                dtype=tf.float32)
            bias = self.add_weight(
                name='bias_0',
                shape=(_hidden_size),
                initializer=self.hidden_b_init)
            self.kernels.append(kernel)
            self.bias.append(bias)
            for i, hidden_size in enumerate(self.hidden_sizes[1:-1]):
                kernel = self.add_weight(
                    name='kernel_{}'.format(i + 1),
                    shape=[_hidden_size, hidden_size],
                    initializer=self.hidden_w_init,
                    dtype=tf.float32)
                bias = self.add_weight(
                    name='bias_{}'.format(i + 1),
                    shape=(_hidden_size),
                    initializer=self.hidden_b_init)
                _hidden_size = hidden_size
                self.kernels.append(kernel)
                self.bias.append(bias)
            kernel = self.add_weight(
                name='kernel_{}'.format(len(self.hidden_sizes)),
                shape=[_hidden_size, self.output_dim],
                initializer=self.output_w_init,
                dtype=tf.float32)
            bias = self.add_weight(
                name='bias_{}'.format(len(self.hidden_sizes)),
                shape=(self.output_dim),
                initializer=self.output_b_init)
            self.kernels.append(kernel)
            self.bias.append(bias)
        super().build(input_shape)

    def call(self, x):
        for kernel, bias in zip(self.kernels[:-1], self.bias[:-1]):
            x = self.hidden_nonlinearity(tf.matmul(x, kernel) + bias) \
                if self.hidden_nonlinearity else x
        x = self.output_nonlinearity(tf.matmul(x, self.kernels[-1]) + self.bias[-1]) \
            if self.output_nonlinearity else x
        return x

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'hidden_sizes': self.hidden_sizes
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":

    custom_layer = MLPLayer(2, (32))
    print("\n### MLPL layer: {}".format(custom_layer.get_config()))

    custom_layer = MLPLayer(2, (32, 32))
    print("\n### MLPL layer: {}".format(custom_layer.get_config()))

    custom_layer = MLPLayer(2, (32, 32, 32))
    print("\n### MLPL layer: {}".format(custom_layer.get_config()))

    model = Sequential()
    model.add(Dense(5, input_dim=2))
    model.add(MLPLayer(2, (32, 32)))

    print("\n### Keras custom model : {}\n".format(model.get_config()))
    print("\n### Keras custom model inputs: {}\n".format(model.inputs))
    print("\n### Keras custom model outputs: {}\n".format(model.outputs))

    print("\nJSON : {}\n".format(model.to_json()))
    fresh_model = tf.keras.models.model_from_json(
        model.to_json(), custom_objects={'MLPLayer': MLPLayer})
    print("\nRestore from JSON : {}\n".format(fresh_model.get_config()))

    print("\nYAML : {}\n".format(model.to_yaml()))
    fresh_model = tf.keras.models.model_from_yaml(
        model.to_yaml(), custom_objects={'MLPLayer': MLPLayer})
    print("\nRestore from YAML: {}\n".format(fresh_model.get_config()))

    print("\n### Saving/loading weights...")
    weights = model.get_weights()
    weights[2] = np.ones_like(weights[2])
    model.set_weights(weights)
    print("\nweights being saved: {}".format(model.get_weights()))
    model.save_weights('./test_keras.weights')

    print("\nweights before load: {}".format(fresh_model.get_weights()))
    fresh_model.load_weights('./test_keras.weights')
    print("\nweights after load: {}".format(fresh_model.get_weights()))

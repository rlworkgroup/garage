"""MLP Layer based on tf.keras.layer."""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import Sequential

# flake8: noqa


class CustomLayer(KerasLayer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return tf.keras.backend.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class CustomModel(KerasModel):
    def __init__(self, num_classes=10):
        super().__init__(name='mlp')
        self.num_classes = num_classes

        # using Keras layers
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


if __name__ == "__main__":

    dense = Dense(10)
    print("\n### Dense layer : {}".format(dense.get_config()))  # wonderful

    custom_layer = CustomLayer(10)
    print("\n### Custom layer: {}".format(
        custom_layer.get_config()))  # basically empty

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=10))
    model.add(Dense(2, activation='softmax'))
    print("\n### Keras model : {}".format(model.get_config()))

    model = Sequential()
    model.add(custom_layer)
    print("\n### Keras custom model : {}\n".format(model.get_config()))

    model = CustomModel(num_classes=2)
    print(model.get_config())
    """
    Console log:

    ### Dense layer : {'name': 'dense', 'trainable': True, 'dtype': None, 'units': 10, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}

    ### Custom layer: {'name': 'custom_layer', 'trainable': True, 'dtype': None}

    ### Keras model : [{'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'batch_input_shape': (None, 10), 'dtype': 'float32', 'units': 32, 'activation': ..., 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}}, {'class_name': 'Dense', 'config': {'name': 'dense_2', 'trainable': True, 'dtype': 'float32', 'units': 2, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None, 'dtype': 'float32'}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {'dtype': 'float32'}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}}]

    ### Keras custom model : [{'class_name': 'CustomLayer', 'config': {'name': 'custom_layer', 'trainable': True, 'dtype': None}}]

    Traceback (most recent call last):
      File "tests/garage/test_keras.py", line 62, in <module>
        print(model.get_config())
      File "/Users/wongtsankwong/miniconda2/envs/garage/lib/python3.6/site-packages/tensorflow/python/keras/engine/network.py", line 971, in get_config
        raise NotImplementedError
    NotImplementedError
    """

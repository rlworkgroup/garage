"""MLP Layer based on tf.keras.layer."""
import numpy as np
import pickle
import tensorflow as tf
from garage.tf.models import GaussianMLPModel2
from garage.tf.core.parameter import parameter
from tests.fixtures import TfGraphTestCase

# flake8: noqa
# pylint: noqa


class TestKerasModel(TfGraphTestCase):
    def test_parameter(self):
        input_var = tf.placeholder(tf.float32, shape=(None, 5))
        parameters = parameter(input_var=input_var, length=10)

        self.sess.run(tf.global_variables_initializer())
        yy = self.sess.run(
            parameters, feed_dict={input_var: np.random.random((2, 5))})

    def test_gaussian_mlp(self):

        model = GaussianMLPModel2(
            input_dim=5, output_dim=2, hidden_sizes=(4, 4))

        self.sess.run(tf.global_variables_initializer())
        data = np.random.random((2, 5))

        y = self.sess.run(model.mean, feed_dict={model.input: data})

        x = pickle.dumps(model)
        model_pickled = pickle.loads(x)

        y = self.sess.run(model.mean, feed_dict={model.input: data})
        y2 = self.sess.run(
            model_pickled.mean, feed_dict={model_pickled.input: data})
        print(y)
        print(y2)

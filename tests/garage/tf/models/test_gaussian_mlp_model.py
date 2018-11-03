import pickle

import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPModel(TfGraphTestCase):
    def test_gaussian_mlp_model(self):
        # Model that does not share networks and adapt std
        model1 = GaussianMLPModel(
            input_dim=2,
            output_dim=2,
            hidden_sizes=[3, 3],
        )

        # Model that share networks
        model2 = GaussianMLPModel(
            input_dim=2,
            output_dim=2,
            hidden_sizes=[3, 3],
            std_share_network=True,
            output_nonlinearity=tf.nn.relu,
        )

        # Model that does not share network but adapt std
        model3 = GaussianMLPModel(
            input_dim=2,
            output_dim=2,
            hidden_sizes=[3, 3],
            adaptive_std=True,
            max_std=1.,
        )

        models = [model1, model2, model3]

        self.sess.run(tf.global_variables_initializer())
        for model in models:
            self.sess.run(
                model.outputs,
                feed_dict={model.inputs: np.zeros(shape=(2, 2))})

    def test_model_pickle(self):
        model = GaussianMLPModel(
            input_dim=2,
            output_dim=2,
            hidden_sizes=[3, 3],
        )
        model_pickled = pickle.loads(pickle.dumps(model))
        self.sess.run(tf.global_variables_initializer())
        data = np.array([[1., 2.]])
        results = self.sess.run(model.outputs, feed_dict={model.inputs: data})
        results_pickled = self.sess.run(
            model_pickled.outputs, feed_dict={model_pickled.inputs: data})

        assert np.all(results["mean"] == results_pickled["mean"])
        assert np.all(results["std"] == results_pickled["std"])

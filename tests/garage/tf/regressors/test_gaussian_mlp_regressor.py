import numpy as np
import tensorflow as tf

from garage.tf.regressors import GaussianMLPRegressor
from tests.fixtures import TfGraphTestCase


class TestGaussianMlpRegressor(TfGraphTestCase):
    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_normalized(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ), output_dim=1)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 1))
        for i in range(150):
            gmr.fit(observations, returns)
            # There will be new assign operations created in the first
            # iteration so let's take the second one to check.
            if i == 1:
                assign_ops_counts = np.sum(
                    np.array([
                        'Assign' in n.name
                        for n in tf.get_default_graph().as_graph_def().node
                    ]).astype(int))
        assign_ops_counts_after = np.sum(
            np.array([
                'Assign' in n.name
                for n in tf.get_default_graph().as_graph_def().node
            ]).astype(int))

        assert assign_ops_counts == assign_ops_counts_after

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }
        prediction = gmr.predict(paths['observations'])
        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

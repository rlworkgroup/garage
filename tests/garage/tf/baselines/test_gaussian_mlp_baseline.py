import numpy as np

from garage.tf.baselines import GaussianMLPBaselineWithModel
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPBaseline(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.box_env = TfEnv(DummyBoxEnv(obs_dim=(2, )))
        self.gmb = GaussianMLPBaselineWithModel(env_spec=self.box_env.spec)

    def tearDown(self):
        super().tearDown()
        self.box_env.close()

    def test_fit(self):
        obs = [{
            'observations': [[0, 0]],
            'returns': [0]
        }, {
            'observations': [[1, 1]],
            'returns': [1]
        }, {
            'observations': [[2, 2]],
            'returns': [2]
        }]

        for _ in range(100):
            self.gmb.fit(obs)

        paths = {'observations': [[0, 0], [1, 1], [2, 2]]}

        prediction = self.gmb.predict(paths)
        assert np.allclose(prediction, [0, 1, 2], rtol=0, atol=1e-3)

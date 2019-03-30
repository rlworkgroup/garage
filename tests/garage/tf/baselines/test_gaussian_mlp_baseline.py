import numpy as np

from garage.tf.baselines import GaussianMLPBaselineWithModel
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPBaseline(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.box_env = TfEnv(DummyBoxEnv(obs_dim=(1, )))
        self.gmb = GaussianMLPBaselineWithModel(env_spec=self.box_env.spec)

    def tearDown(self):
        super().tearDown()
        self.box_env.close()

    def test_fit(self):
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        for _ in range(100):
            self.gmb.fit(obs)

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }
        prediction = self.gmb.predict(paths)

        expected = [0, -1, -0.707, 0, 0.707, 1, 0]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

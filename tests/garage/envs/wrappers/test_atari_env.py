import numpy as np

from garage.envs.wrappers import AtariEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnvBaselines


class TestFireReset:
    def test_atari_env(self):
        env = DummyDiscretePixelEnvBaselines()
        env_wrapped = AtariEnv(env)
        obs = env.reset()
        obs_wrapped = env_wrapped.reset()
        assert not isinstance(obs, np.ndarray)
        assert isinstance(obs_wrapped, np.ndarray)

        obs, _, _, _ = env.step(1)
        obs_wrapped, _, _, _ = env_wrapped.step(1)
        assert not isinstance(obs, np.ndarray)
        assert isinstance(obs_wrapped, np.ndarray)

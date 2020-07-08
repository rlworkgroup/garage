import numpy as np

from garage.envs.wrappers import EpisodicLife

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestEpisodicLife:

    def test_episodic_life_reset(self):
        env = EpisodicLife(DummyDiscretePixelEnv())
        obs = env.reset()

        # env has reset
        assert np.array_equal(obs, np.ones(env.observation_space.shape))
        assert env.unwrapped.ale.lives() == 5

        obs, _, d, info = env.step(0)
        assert d
        assert info['ale.lives'] == 4
        obs = env.reset()

        # env has not reset
        assert not np.array_equal(obs, np.ones(env.observation_space.shape))

        for _ in range(3):
            obs, _, d, info = env.step(0)
        assert d
        assert info['ale.lives'] == 0
        obs = env.reset()
        # env has reset
        assert np.array_equal(obs, np.ones(env.observation_space.shape))

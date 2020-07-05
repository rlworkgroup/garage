import numpy as np

from garage.envs.wrappers import MaxAndSkip

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestMaxAndSkip:

    def setup_method(self):
        self.env = DummyDiscretePixelEnv(random=False)
        self.env_wrap = MaxAndSkip(DummyDiscretePixelEnv(random=False), skip=4)

    def teardown_method(self):
        self.env.close()
        self.env_wrap.close()

    def test_max_and_skip_reset(self):
        np.testing.assert_array_equal(self.env.reset(), self.env_wrap.reset())

    def test_max_and_skip_step(self):
        self.env.reset()
        self.env_wrap.reset()
        obs_wrap, reward_wrap, _, _ = self.env_wrap.step(1)
        reward = 0
        for _ in range(4):
            obs, r, _, _ = self.env.step(1)
            reward += r

        np.testing.assert_array_equal(obs, obs_wrap)
        np.testing.assert_array_equal(reward, reward_wrap)

        # done=True because both env stepped more than 4 times in total
        obs_wrap, _, done_wrap, _ = self.env_wrap.step(1)
        obs, _, done, _ = self.env.step(1)

        assert done
        assert done_wrap
        np.testing.assert_array_equal(obs, obs_wrap)

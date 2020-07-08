from garage.envs.wrappers import ClipReward

from tests.fixtures.envs.dummy import DummyRewardBoxEnv


class TestClipReward:

    def test_clip_reward(self):
        # reward = 10 when action = 0, otherwise -10
        env = DummyRewardBoxEnv(random=True)
        env_wrap = ClipReward(env)
        env.reset()
        env_wrap.reset()

        _, reward, _, _ = env.step(0)
        _, reward_wrap, _, _ = env_wrap.step(0)

        assert reward == 10
        assert reward_wrap == 1

        _, reward, _, _ = env.step(1)
        _, reward_wrap, _, _ = env_wrap.step(1)

        assert reward == -10
        assert reward_wrap == -1

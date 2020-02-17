from garage.envs import PointEnv
from garage.envs import RL2Env


class TestRL2Env:

    # pylint: disable=unsubscriptable-object
    def test_observation_dimension(self):
        env = PointEnv()
        wrapped_env = RL2Env(PointEnv())
        assert wrapped_env.spec.observation_space.shape[0] == (
            env.observation_space.shape[0] + env.action_space.shape[0] + 2)
        obs = env.reset()
        obs2 = wrapped_env.reset()
        assert obs.shape[0] + env.action_space.shape[0] + 2 == obs2.shape[0]
        obs, _, _, _ = env.step(env.action_space.sample())
        obs2, _, _, _ = wrapped_env.step(env.action_space.sample())
        assert obs.shape[0] + env.action_space.shape[0] + 2 == obs2.shape[0]

    # pylint: disable=unsubscriptable-object
    def test_observation_dimension_with_max_obs_dim(self):
        env = PointEnv()
        wrapped_env = RL2Env(PointEnv(), max_obs_dim=10)
        assert wrapped_env.spec.observation_space.shape[
            0] == 10 + env.action_space.shape[0] + 2
        obs = wrapped_env.reset()
        assert 10 + env.action_space.shape[0] + 2 == obs.shape[0]
        obs, _, _, _ = wrapped_env.step(env.action_space.sample())
        assert 10 + env.action_space.shape[0] + 2 == obs.shape[0]

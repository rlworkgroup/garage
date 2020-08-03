from garage.envs import PointEnv
from garage.tf.algos.rl2 import RL2Env


class TestRL2Env:

    # pylint: disable=unsubscriptable-object
    def test_observation_dimension(self):
        env = PointEnv()
        wrapped_env = RL2Env(PointEnv())
        assert wrapped_env.spec.observation_space.shape[0] == (
            env.observation_space.shape[0] + env.action_space.shape[0] + 2)
        obs, _ = env.reset()
        obs2, _ = wrapped_env.reset()
        assert obs.shape[0] + env.action_space.shape[0] + 2 == obs2.shape[0]
        obs = env.step(env.action_space.sample()).observation
        obs2 = wrapped_env.step(env.action_space.sample()).observation
        assert obs.shape[0] + env.action_space.shape[0] + 2 == obs2.shape[0]

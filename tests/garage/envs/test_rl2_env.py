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

    def test_step(self):
        env = RL2Env(PointEnv())

        env.reset()
        es = env.step(env.action_space.sample())
        assert env.observation_space.contains(es.observation)

    def test_visualization(self):
        env = PointEnv()
        wrapped_env = RL2Env(env)

        assert env.render_modes == wrapped_env.render_modes
        mode = env.render_modes[0]
        assert env.render(mode) == wrapped_env.render(mode)

        wrapped_env.reset()
        wrapped_env.visualize()
        wrapped_env.step(wrapped_env.action_space.sample())
        wrapped_env.close()

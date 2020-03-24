import pytest

from garage.envs import EnvSpec, GarageEnv


class TestGarageEnv:

    def test_wraps_env_spec(self):
        garage_env = GarageEnv(env_name='Pendulum-v0')
        assert isinstance(garage_env.spec, EnvSpec)

    def test_closes_box2d(self):
        garage_env = GarageEnv(env_name='CarRacing-v0')
        garage_env.render()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    @pytest.mark.mujoco
    def test_closes_mujoco(self):
        garage_env = GarageEnv(env_name='Ant-v2')
        garage_env.render()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    def test_time_limit_env(self):
        garage_env = GarageEnv(env_name='Pendulum-v0')
        garage_env.reset()
        for _ in range(200):
            _, _, done, info = garage_env.step(
                garage_env.spec.action_space.sample())
        assert not done and info['TimeLimit.truncated']
        assert info['GarageEnv.TimeLimitTerminated']

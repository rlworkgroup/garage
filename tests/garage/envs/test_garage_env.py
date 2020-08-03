import gym
import pytest

from garage.envs import EnvSpec, GarageEnv
from garage.envs.bullet import BulletEnv


class TestGarageEnv:

    def test_wraps_env_spec(self):
        garage_env = GarageEnv(env_name='Pendulum-v0')
        assert isinstance(garage_env.spec, EnvSpec)

    def test_closes_box2d(self):
        garage_env = GarageEnv(env_name='CarRacing-v0')
        garage_env.visualize()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    @pytest.mark.mujoco
    def test_closes_mujoco(self):
        garage_env = GarageEnv(env_name='Ant-v2')
        garage_env.visualize()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    def test_time_limit_env(self):
        garage_env = GarageEnv(env_name='Pendulum-v0', max_episode_length=200)
        garage_env.reset()
        for _ in range(200):
            ts = garage_env.step(garage_env.spec.action_space.sample())
        assert ts.timeout and ts.env_info['TimeLimit.truncated']
        assert ts.env_info['GarageEnv.TimeLimitTerminated']

    def test_return_bullet_env(self):
        env = GarageEnv(env=gym.make('CartPoleBulletEnv-v1'))
        assert isinstance(env, BulletEnv)
        env = GarageEnv(env_name='CartPoleBulletEnv-v1')
        assert isinstance(env, BulletEnv)
        env = GarageEnv(gym.make('CartPoleBulletEnv-v1'))
        assert isinstance(env, BulletEnv)
        env = GarageEnv(None, 'CartPoleBulletEnv-v1')
        assert isinstance(env, BulletEnv)

import pickle

import gym
import pybullet_envs
import pytest

from garage.envs import GarageEnv
from tests.helpers import step_env_with_gym_quirks


class TestGarageEnv:

    def test_is_pickleable(self):
        env = GarageEnv(env_name='CartPole-v1')
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        bullet_envs = [
            env.replace('- ', '') for env in pybullet_envs.getList()
        ]
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        elif spec._env_name in bullet_envs:
            pytest.skip('Bullet environments will be loaded differently via '
                        'BulletEnv()')
        env = GarageEnv(spec.make())
        step_env_with_gym_quirks(env, spec)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        bullet_envs = [
            env.replace('- ', '') for env in pybullet_envs.getList()
        ]
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        elif spec._env_name in bullet_envs:
            pytest.skip('Bullet environments will be loaded differently via '
                        'BulletEnv()')
        env = GarageEnv(env_name=spec.id)
        step_env_with_gym_quirks(env,
                                 spec,
                                 n=1,
                                 render=True,
                                 serialize_env=True)

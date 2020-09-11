import pickle

import gym
import pytest

from garage.envs import GarageEnv
from tests.helpers import step_env_with_gym_quirks


class TestGarageEnv:

    def test_is_pickleable(self):
        env = GarageEnv(env_name='CartPole-v1')
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.spec == env.spec
        assert round_trip.env.spec.id == env.env.spec.id
        assert (round_trip.env.spec.max_episode_steps ==
                env.env.spec.max_episode_steps)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        env = GarageEnv(spec.make())
        step_env_with_gym_quirks(env, spec)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        if spec.id == 'KellyCoinflipGeneralized-v0':
            pytest.skip(
                'KellyCoinflipGeneralized-v0\'s action space is random')
        env = GarageEnv(env_name=spec.id)
        step_env_with_gym_quirks(env,
                                 spec,
                                 n=1,
                                 render=True,
                                 serialize_env=True)

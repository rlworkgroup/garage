import pickle

import gym
import pytest

from garage.envs import GymEnv

from tests.helpers import step_env_with_gym_quirks


class TestGymEnv:

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        if spec._env_name.startswith('CarRacing'):
            pytest.skip(
                'CarRacing-* envs bundled in atari-py 0.2.x don\'t load')
        if 'Kuka' in spec.id:
            # Kuka environments calls py_bullet.resetSimulation() in reset()
            # unconditionally, which globally resets other simulations. So
            # only one Kuka environment can be tested.
            pytest.skip('Skip Kuka Bullet environments')
        env = GymEnv(spec.id)
        step_env_with_gym_quirks(env, spec, visualize=False)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        if 'Kuka' in spec.id:
            # Kuka environments calls py_bullet.resetSimulation() in reset()
            # unconditionally, which globally resets other simulations. So
            # only one Kuka environment can be tested.
            pytest.skip('Skip Kuka Bullet environments')
        elif 'Minitaur' in spec.id:
            pytest.skip('Bulle Minitaur envs don\'t load')
        env = GymEnv(spec.id)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip

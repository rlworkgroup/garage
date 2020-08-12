"""Test module for BulletEnv"""
import pickle

import pybullet_envs
from pybullet_utils.bullet_client import BulletClient
import pytest

from garage.envs.bullet import _get_unsupported_env_list, BulletEnv

from tests.helpers import step_env


@pytest.mark.parametrize('env_ids', [pybullet_envs.getList()])
def test_can_step(env_ids):
    """Test Bullet environments can step"""

    for env_id in env_ids:
        # extract id string
        env_id = env_id.replace('- ', '')
        if env_id == 'KukaCamBulletEnv-v0':
            # Kuka environments calls py_bullet.resetSimulation() in reset()
            # unconditionally, which globally resets other simulations. So
            # only one Kuka environment is tested.
            continue
        if env_id in _get_unsupported_env_list():
            pytest.skip('Skip unsupported Bullet environments')
        env = BulletEnv(env_id)
        ob_space = env.observation_space
        act_space = env.action_space
        env.reset()

        ob = ob_space.sample()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(env, visualize=False)
        env.close()


@pytest.mark.parametrize('env_ids', [pybullet_envs.getList()])
def test_pickleable(env_ids):
    """Test Bullet environments are pickle-able"""
    for env_id in env_ids:
        # extract id string
        env_id = env_id.replace('- ', '')
        if env_id in _get_unsupported_env_list():
            pytest.skip('Skip unsupported Bullet environments')
        env = BulletEnv(env_id)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        env.close()


@pytest.mark.parametrize('env_ids', [pybullet_envs.getList()])
def test_pickle_creates_new_server(env_ids):
    """Test pickling a Bullet environment creates a new connection.

    If all pickling create new connections, no repetition of client id
    should be found.
    """
    n_env = 4
    for env_id in env_ids:
        # extract id string
        env_id = env_id.replace('- ', '')
        if env_id in _get_unsupported_env_list():
            pytest.skip('Skip unsupported Bullet environments')
        bullet_env = BulletEnv(env_id)
        envs = [pickle.loads(pickle.dumps(bullet_env)) for _ in range(n_env)]
        id_set = set()

        if hasattr(bullet_env._env, '_pybullet_client'):
            id_set.add(bullet_env._env._pybullet_client._client)
            for e in envs:
                new_id = e._env._pybullet_client._client
                assert new_id not in id_set
                id_set.add(new_id)
        elif hasattr(bullet_env._env, '_p'):
            if isinstance(bullet_env._env._p, BulletClient):
                id_set.add(bullet_env._env._p._client)
                for e in envs:
                    new_id = e._env._p._client
                    assert new_id not in id_set
                    id_set.add(new_id)
            else:
                # Some environments have _p as the pybullet module, and they
                # don't store client id, so can't check here
                pass

        for env in envs:
            env.close()


def test_time_limit_env():
    """Test BulletEnv emits done signal when time limit expiration occurs.

    After setting max_episode_steps=50, info['GymEnv.TimeLimitTerminated']
    is expected to be True after 50 steps.

    """
    env = BulletEnv('MinitaurBulletEnv-v0')
    env._env._max_episode_steps = 50
    env.reset()
    for _ in range(50):
        es = env.step(env.spec.action_space.sample())
    assert not es.terminal and es.env_info['TimeLimit.truncated']
    assert es.env_info['GymEnv.TimeLimitTerminated']

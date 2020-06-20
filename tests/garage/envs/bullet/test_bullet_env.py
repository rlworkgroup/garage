"""Test module for BulletEnv"""
import pickle

from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
from pybullet_envs.bullet.cartpole_bullet import CartPoleContinuousBulletEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.minitaur_duck_gym_env import MinitaurBulletDuckEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_utils.bullet_client import BulletClient
import pytest

from garage.envs.bullet import BulletEnv
from tests.helpers import step_env


@pytest.mark.parametrize('envs', [[
    CartPoleBulletEnv, CartPoleContinuousBulletEnv, MinitaurBulletEnv,
    MinitaurBulletDuckEnv, RacecarGymEnv, RacecarZEDGymEnv, KukaGymEnv
]])
def test_can_step(envs):
    """Test Pybullet environments can step"""
    for pybullet_env in envs:
        if pybullet_env == RacecarZEDGymEnv:
            env = BulletEnv(pybullet_env(renders=False))
        else:
            env = BulletEnv(pybullet_env())
        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(env, render=False)
        env.close()


@pytest.mark.parametrize('envs', [[
    CartPoleBulletEnv, CartPoleContinuousBulletEnv, MinitaurBulletEnv,
    MinitaurBulletDuckEnv, RacecarGymEnv, RacecarZEDGymEnv, KukaGymEnv
]])
def test_pickleable(envs):
    """Test Pybullet environments are pickleable"""
    for pybullet_env in envs:
        if pybullet_env == RacecarZEDGymEnv:
            env = BulletEnv(pybullet_env(renders=False))
        else:
            env = BulletEnv(pybullet_env())
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip


@pytest.mark.parametrize('envs', [[
    CartPoleBulletEnv, CartPoleContinuousBulletEnv, MinitaurBulletEnv,
    MinitaurBulletDuckEnv, RacecarGymEnv, RacecarZEDGymEnv, KukaGymEnv
]])
def test_pickle_creates_new_server(envs):
    """Test pickleing a Pybullet environment creates a new connection.

    If all pickleing create new connections, no repetition of client id
    should be found.
    """
    n_env = 4
    for pybullet_env in envs:
        if pybullet_env == RacecarZEDGymEnv:
            env = BulletEnv(pybullet_env(renders=False))
        else:
            env = BulletEnv(pybullet_env())
        envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_env)]
        id_set = set()

        if hasattr(pybullet_env, '_pybullet_client'):
            id_set.add(pybullet_env._pybullet_client._client)
            for e in envs:
                new_id = e._env._pybullet_client._client
                assert new_id not in id_set
                id_set.add(new_id)
        elif hasattr(pybullet_env, '_p'):
            if isinstance(pybullet_env._p, BulletClient):
                id_set.add(pybullet_env._p._client)
                for e in envs:
                    new_id = e._env._p._client
                    assert new_id not in id_set
                    id_set.add(new_id)
            else:
                # Kuka environment has _p as the pybullet module.
                # Since Kuka env doesn't store client id, so can't check
                # here
                pass

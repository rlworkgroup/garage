# yapf: disable
import functools
import unittest.mock

import numpy as np
import pytest

from garage.experiment import task_sampler

# yapf: enable

try:
    # pylint: disable=unused-import
    import mujoco_py  # noqa: F401
except ImportError:
    pytest.skip('To use mujoco-based features, please install garage[mujoco].',
                allow_module_level=True)
except Exception:  # pylint: disable=broad-except
    pytest.skip(
        'Skipping tests, failed to import mujoco. Do you have a '
        'valid mujoco key installed?',
        allow_module_level=True)

from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv  # isort:skip # noqa: E501


def test_env_pool_sampler():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    envs = [HalfCheetahVelEnv() for _ in range(5)]
    tasks = task_sampler.EnvPoolSampler(envs)
    assert tasks.n_tasks == 5
    updates = tasks.sample(5)
    for env in envs:
        assert any(env is update() for update in updates)
    with pytest.raises(ValueError):
        tasks.sample(5, with_replacement=True)
    with pytest.raises(ValueError):
        tasks.sample(6)
    tasks.grow_pool(10)
    tasks.sample(10)


def test_construct_envs_sampler():
    env_constructors = [HalfCheetahVelEnv for _ in range(5)]
    tasks = task_sampler.ConstructEnvsSampler(env_constructors)
    assert tasks.n_tasks == 5
    updates = tasks.sample(15)
    envs = [update() for update in updates]
    action = envs[0].action_space.sample()
    rewards = [env.step(action)[1] for env in envs]
    assert np.var(rewards) > 0
    env = envs[0]
    env.close = unittest.mock.MagicMock(name='env.close')
    updates[-1](env)
    env.close.assert_called()


@pytest.mark.mujoco
def test_set_task_task_sampler_ml10():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    ml10 = metaworld.ML10()
    tasks = task_sampler.MetaWorldTaskSampler(ml10, 'test')
    assert tasks.n_tasks == 5 * 50
    with pytest.raises(ValueError):
        tasks.sample(1)
    updates = tasks.sample(10)
    envs = [update() for update in updates]
    for env in envs:
        env.reset()
    action = envs[0].action_space.sample()
    rewards = [env.step(action).reward for env in envs]
    assert np.var(rewards) > 0
    env = envs[0]
    env.close = unittest.mock.MagicMock(name='env.close')
    updates[1](env)
    env.close.assert_not_called()
    updates[2](env)
    env.close.assert_called()


@pytest.mark.mujoco
def test_set_task_task_sampler_half_cheetah_vel_env():
    tasks = task_sampler.SetTaskSampler(HalfCheetahVelEnv)
    assert tasks.n_tasks is None
    updates = tasks.sample(10)
    envs = [update() for update in updates]
    action = envs[0].action_space.sample()
    rewards = [env.step(action)[1] for env in envs]
    assert np.var(rewards) > 0
    env = envs[0]
    env.close = unittest.mock.MagicMock(name='env.close')
    updates[-1](env)
    env.close.assert_not_called()


@pytest.mark.mujoco
def test_metaworld_sample_and_step():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    ml1 = metaworld.ML1('push-v1')
    tasks = task_sampler.MetaWorldTaskSampler(ml1, 'train')
    updates = tasks.sample(100)
    assert len(updates) == 100
    env = updates[0]()
    action = env.action_space.sample()
    env.reset()
    env.step(action)
    env.step(action)
    env.close()
    updates = tasks.sample(100, with_replacement=True)
    assert len(updates) == 100

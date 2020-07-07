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


@pytest.mark.mujoco
def test_env_pool_sampler():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    from metaworld.benchmarks import ML10
    train_tasks = ML10.get_train_tasks().all_task_names
    ML10_train_envs = [
        ML10.from_task(train_task) for train_task in train_tasks
    ]
    tasks = task_sampler.EnvPoolSampler(ML10_train_envs)
    assert tasks.n_tasks == 10
    updates = tasks.sample(10)
    for env in ML10_train_envs:
        assert any(env is update() for update in updates)
    with pytest.raises(ValueError):
        tasks.sample(10, with_replacement=True)
    with pytest.raises(ValueError):
        tasks.sample(11)
    tasks.grow_pool(20)
    tasks.sample(20)


@pytest.mark.mujoco
def test_construct_envs_sampler_ml10():
    # pylint: disable=import-outside-toplevel
    from metaworld.benchmarks import ML10
    train_tasks = ML10.get_train_tasks().all_task_names
    ML10_constructors = [
        functools.partial(ML10.from_task, train_task)
        for train_task in train_tasks
    ]
    tasks = task_sampler.ConstructEnvsSampler(ML10_constructors)
    assert tasks.n_tasks == 10
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
    # pylint: disable=import-outside-toplevel
    from metaworld.benchmarks import ML10
    tasks = task_sampler.SetTaskSampler(ML10.get_train_tasks)
    assert tasks.n_tasks == 10
    updates = tasks.sample(3)
    envs = [update() for update in updates]
    action = envs[0].action_space.sample()
    rewards = [env.step(action)[1] for env in envs]
    assert np.var(rewards) > 0
    env = envs[0]
    env.close = unittest.mock.MagicMock(name='env.close')
    updates[-1](env)
    env.close.assert_not_called()


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

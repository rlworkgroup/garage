import functools
import unittest.mock

import numpy as np
import pytest

from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler


@pytest.mark.large
def test_env_pool_sampler():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
    from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT
    ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
    ML10_ENVS = MEDIUM_MODE_CLS_DICT

    ML10_train_envs = [
        env(*ML10_ARGS['train'][task]['args'],
            **ML10_ARGS['train'][task]['kwargs'])
        for (task, env) in ML10_ENVS['train'].items()
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


@pytest.mark.large
def test_construct_envs_sampler_ml10():
    # pylint: disable=import-outside-toplevel
    from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
    from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT
    ML10_ARGS = MEDIUM_MODE_ARGS_KWARGS
    ML10_ENVS = MEDIUM_MODE_CLS_DICT

    ML10_constructors = [
        functools.partial(env, *ML10_ARGS['train'][task]['args'],
                          **ML10_ARGS['train'][task]['kwargs'])
        for (task, env) in ML10_ENVS['train'].items()
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


@pytest.mark.large
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

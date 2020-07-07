import pickle

import pytest

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

from garage.envs.mujoco.half_cheetah_dir_env import HalfCheetahDirEnv  # isort:skip # noqa: E501
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv  # isort:skip # noqa: E501


@pytest.mark.mujoco
@pytest.mark.parametrize('env_type', [HalfCheetahVelEnv, HalfCheetahDirEnv])
def test_can_sim(env_type):
    env = env_type()
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    for _ in range(3):
        env.step(env.action_space.sample())


@pytest.mark.mujoco
@pytest.mark.parametrize('env_type', [HalfCheetahVelEnv, HalfCheetahDirEnv])
def test_pickling_keeps_goal(env_type):
    env = env_type()
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    env_clone = pickle.loads(pickle.dumps(env))
    assert env._task == env_clone._task

import math

import gym
import pytest

from garage import EnvSpec
from garage.envs import GymEnv
from garage.envs.bullet import BulletEnv
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames


def test_wraps_env_spec():
    garage_env = GymEnv('Pendulum-v0')
    assert isinstance(garage_env.spec, EnvSpec)


def test_closes_box2d():
    garage_env = GymEnv('CarRacing-v0')
    garage_env.visualize()
    assert garage_env._env.viewer is not None
    garage_env.close()
    assert garage_env._env.viewer is None


@pytest.mark.mujoco
def test_closes_mujoco():
    garage_env = GymEnv('Ant-v2')
    garage_env.visualize()
    assert garage_env._env.viewer is not None
    garage_env.close()
    assert garage_env._env.viewer is None


def test_time_limit_env():
    garage_env = GymEnv('Pendulum-v0', max_episode_length=200)
    garage_env._env._max_episode_steps = 200
    garage_env.reset()
    for _ in range(200):
        es = garage_env.step(garage_env.spec.action_space.sample())
    assert es.timeout and es.env_info['TimeLimit.truncated']
    assert es.env_info['GymEnv.TimeLimitTerminated']


def test_process_env_argument():
    env = GymEnv(env=gym.make('MountainCar-v0'))
    env.close()
    env = GymEnv(env='MountainCar-v0')
    env.close()
    env = GymEnv(gym.make('MountainCar-v0'))
    env.close()
    env = GymEnv('MountainCar-v0')
    env.close()
    with pytest.raises(ValueError, match='GymEnv can take env'):
        env = GymEnv(1)
    env.close()


def test_return_bullet_env():
    env = GymEnv(env=gym.make('CartPoleBulletEnv-v1'))
    assert isinstance(env, BulletEnv)
    env = GymEnv(env='CartPoleBulletEnv-v1')
    assert isinstance(env, BulletEnv)
    env = GymEnv(gym.make('CartPoleBulletEnv-v1'))
    assert isinstance(env, BulletEnv)
    env = GymEnv('CartPoleBulletEnv-v1')
    assert isinstance(env, BulletEnv)


def test_get_time_limit_from_gym_make():
    env = GymEnv('CartPoleBulletEnv-v1')
    assert env._max_episode_length == env._env.spec.max_episode_steps
    # verify it matches TimeLimit._max_episode_steps
    assert env._max_episode_length == env._env._max_episode_steps

    max_steps_arg = 1
    spec = list(gym.envs.registry.all())[0]
    env = GymEnv(spec.make())
    assert env._max_episode_length == env._env.spec.max_episode_steps

    spec.max_episode_steps = None
    env = GymEnv(spec.make(), max_episode_length=max_steps_arg)
    assert env._max_episode_length == max_steps_arg

    env = GymEnv(spec.make(), max_episode_length=None)
    assert env._max_episode_length is None


def test_get_time_limit_detects_inconsistency():
    with pytest.raises(RuntimeError):
        env = GymEnv('CartPoleBulletEnv-v1', max_episode_length=math.inf)
        env.close()


def test_get_time_limit_finds_time_limit():
    env = gym.make('PongNoFrameskip-v4')
    time_limit = env._max_episode_steps

    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = ClipReward(env)
    env = StackFrames(env, 4)

    env = GymEnv(env)
    assert env._max_episode_length == time_limit


def test_visualization():
    inner_env = gym.make('MountainCar-v0')
    env = GymEnv(inner_env)

    env.reset()
    env.visualize()
    assert inner_env.metadata['render.modes'] == env.render_modes
    env.step(env.action_space.sample())


def test_catch_no_reset():
    env = GymEnv('MountainCar-v0')
    with pytest.raises(RuntimeError, match='reset()'):
        env.step(1)


def test_done_resets_step_cnt():
    env = GymEnv('MountainCar-v0')
    max_episode_length = env.spec.max_episode_length

    env.reset()
    for _ in range(max_episode_length):
        es = env.step(env.action_space.sample())
        if es.last:
            break
    assert env._step_cnt is None

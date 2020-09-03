from gym.wrappers import TimeLimit
import numpy as np
import pytest

from garage.envs import EnvSpec, GarageEnv
from garage.envs.grid_world_env import GridWorldEnv
from garage.np.policies import ScriptedPolicy


class TestGarageEnv:

    def test_wraps_env_spec(self):
        garage_env = GarageEnv(env_name='Pendulum-v0')
        assert isinstance(garage_env.spec, EnvSpec)

    def test_closes_box2d(self):
        garage_env = GarageEnv(env_name='CarRacing-v0')
        garage_env.render()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    @pytest.mark.mujoco
    def test_closes_mujoco(self):
        garage_env = GarageEnv(env_name='Ant-v2')
        garage_env.render()
        assert garage_env.env.viewer is not None
        garage_env.close()
        assert garage_env.env.viewer is None

    def test_time_limit_env(self):
        garage_env = GarageEnv(env_name='Pendulum-v0')
        garage_env.reset()
        for _ in range(200):
            _, _, done, info = garage_env.step(
                garage_env.spec.action_space.sample())
        assert not done and info['TimeLimit.truncated']
        assert info['GarageEnv.TimeLimitTerminated']


def test_garage_env_idempotent():
    # test if garage env can wrap itself
    env_no_wrap = GridWorldEnv(desc='4x4')
    env_single_wrap = GarageEnv(GridWorldEnv(desc='4x4'))
    env_double_wrap = GarageEnv(GarageEnv(GridWorldEnv(desc='4x4')))

    policy = ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
    obs_nw = env_no_wrap.reset()
    obs_sw = env_single_wrap.reset()
    obs_dw = env_double_wrap.reset()

    for _ in range(16):
        assert np.all(np.equal(obs_nw, obs_sw))
        assert np.all(np.equal(obs_nw, obs_dw))
        assert np.all(np.equal(obs_sw, obs_dw))
        step_nw = env_no_wrap.step(policy.get_action(obs_nw)[0])
        step_sw = env_single_wrap.step(policy.get_action(obs_sw)[0])
        step_dw = env_double_wrap.step(policy.get_action(obs_dw)[0])
        obs_nw = step_nw[0]
        obs_sw = step_sw[0]
        obs_dw = step_dw[0]
        # test that single wrapped and double wrapped envs return the same
        # values
        assert np.all(np.equal(step_sw, step_dw))


def test_garage_env_idempotent_time_limit():
    # test if garage env can wrap itself if environments
    # are wrapped with timelimits
    env_single_wrap = GarageEnv(
        TimeLimit(GridWorldEnv(desc='4x4'), max_episode_steps=16))
    env_double_wrap = GarageEnv(
        GarageEnv(TimeLimit(GridWorldEnv(desc='4x4'), max_episode_steps=16)))
    # purposefully greater than the max path length to expose
    # time limit truncations
    num_steps = 20
    policy = ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
    obs_sw = env_single_wrap.reset()
    obs_dw = env_double_wrap.reset()
    assert np.all(np.equal(obs_sw, obs_dw))
    for _ in range(num_steps):
        step_sw = env_single_wrap.step(policy.get_action(obs_sw)[0])
        step_dw = env_double_wrap.step(policy.get_action(obs_dw)[0])
        obs_sw = step_sw[0]
        obs_dw = step_dw[0]
        # test that single wrapped and double wrapped envs return the same
        # values
        assert np.all(np.equal(step_sw, step_dw))

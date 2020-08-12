import akro
import cloudpickle
import gym
import numpy as np
import pytest

from garage import EnvSpec, EnvStep, StepType


def test_env_spec_pickleable():
    env_spec = EnvSpec(akro.Box(-1, 1, (1, )), akro.Box(-2, 2, (2, )), 500)
    round_trip = cloudpickle.loads(cloudpickle.dumps(env_spec))
    assert round_trip
    assert round_trip.action_space == env_spec.action_space
    assert round_trip.observation_space == env_spec.observation_space
    assert round_trip.max_episode_length == env_spec.max_episode_length


@pytest.fixture
def sample_data():
    # spaces
    obs_space = gym.spaces.Box(low=1,
                               high=10,
                               shape=(4, 3, 2),
                               dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)

    # generate data
    next_obs = obs_space.sample()
    act = act_space.sample()
    rew = 10.0
    step_type = StepType.FIRST

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.array([[1, 1]])
    env_infos['TimeLimit.truncated'] = (step_type == StepType.TIMEOUT)

    return {
        'env_spec': env_spec,
        'observation': next_obs,
        'action': act,
        'reward': rew,
        'env_info': env_infos,
        'step_type': step_type
    }


def test_new_env_step(sample_data):
    s = EnvStep(**sample_data)
    assert s.env_spec is sample_data['env_spec']
    assert s.observation is sample_data['observation']
    assert s.action is sample_data['action']
    assert s.reward is sample_data['reward']
    assert s.step_type is sample_data['step_type']
    assert s.env_info is sample_data['env_info']
    del s

    obs_space = akro.Box(low=-1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = akro.Box(low=-1, high=10, shape=(4, 2), dtype=np.float32)
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec
    obs_space = akro.Box(low=-1000,
                         high=1000,
                         shape=(4, 3, 2),
                         dtype=np.float32)
    act_space = akro.Box(low=-1000, high=1000, shape=(4, 2), dtype=np.float32)
    sample_data['observation'] = obs_space.sample()
    sample_data['action'] = act_space.sample()
    s = EnvStep(**sample_data)

    assert s.observation is sample_data['observation']
    assert s.action is sample_data['action']


def test_step_type_property_env_step(sample_data):
    sample_data['step_type'] = StepType.FIRST
    s = EnvStep(**sample_data)
    assert s.first

    sample_data['step_type'] = StepType.MID
    s = EnvStep(**sample_data)
    assert s.mid

    sample_data['step_type'] = StepType.TERMINAL
    s = EnvStep(**sample_data)
    assert s.terminal and s.last

    sample_data['step_type'] = StepType.TIMEOUT
    s = EnvStep(**sample_data)
    assert s.timeout and s.last

import gym.spaces
import numpy as np
import pytest

from garage import TrajectoryBatch
from garage.envs import EnvSpec


@pytest.fixture
def traj_data():
    # spaces
    obs_space = gym.spaces.Box(low=1,
                               high=np.inf,
                               shape=(4, 3, 2),
                               dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)

    # generate data
    lens = np.array([10, 20, 7, 25, 25, 40, 10, 5])
    n_t = lens.sum()
    obs = np.stack([obs_space.low] * n_t)
    act = np.stack([[1, 3]] * n_t)
    rew = np.arange(n_t)
    terms = np.zeros(n_t, dtype=np.bool)
    terms[np.cumsum(lens) - 1] = True  # set terminal bits

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.stack([[1, 1]] * n_t)
    env_infos['foo'] = np.arange(n_t)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act
    agent_infos['hidden'] = np.arange(n_t)

    return {
        'env_spec': env_spec,
        'observations': obs,
        'actions': act,
        'rewards': rew,
        'terminals': terms,
        'env_infos': env_infos,
        'agent_infos': agent_infos,
        'lengths': lens,
    }


def test_new(traj_data):
    t = TrajectoryBatch(**traj_data)
    assert t.env_spec is traj_data['env_spec']
    assert t.observations is traj_data['observations']
    assert t.actions is traj_data['actions']
    assert t.rewards is traj_data['rewards']
    assert t.terminals is traj_data['terminals']
    assert t.env_infos is traj_data['env_infos']
    assert t.agent_infos is traj_data['agent_infos']
    assert t.lengths is traj_data['lengths']


def test_lengths_shape_mismatch(traj_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must be a tensor of shape'):
        traj_data['lengths'] = traj_data['lengths'].reshape((4, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_lengths_dtype_mismatch(traj_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must have an integer dtype'):
        traj_data['lengths'] = traj_data['lengths'].astype(np.float32)
        t = TrajectoryBatch(**traj_data)
        del t


def test_obs_env_spec_mismatch(traj_data):
    with pytest.raises(ValueError, match='observations must conform'):
        traj_data['observations'] = traj_data['observations'][:, :, :, :1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_obs_batch_mismatch(traj_data):
    with pytest.raises(ValueError, match='batch dimension of observations'):
        traj_data['observations'] = traj_data['observations'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_act_env_spec_mismatch(traj_data):
    with pytest.raises(ValueError, match='actions must conform'):
        traj_data['actions'] = traj_data['actions'][:, 0]
        t = TrajectoryBatch(**traj_data)
        del t


def test_act_batch_mismatch(traj_data):
    with pytest.raises(ValueError, match='batch dimension of actions'):
        traj_data['actions'] = traj_data['actions'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_rewards_shape_mismatch(traj_data):
    with pytest.raises(ValueError, match='Rewards tensor'):
        traj_data['rewards'] = traj_data['rewards'].reshape((2, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_terminals_shape_mismatch(traj_data):
    with pytest.raises(ValueError, match='Terminals tensor must have shape'):
        traj_data['terminals'] = traj_data['terminals'].reshape((2, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_terminals_dtype_mismatch(traj_data):
    with pytest.raises(ValueError, match='Terminals tensor must be dtype'):
        traj_data['terminals'] = traj_data['terminals'].astype(np.float32)
        t = TrajectoryBatch(**traj_data)
        del t


def test_env_infos_not_ndarray(traj_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must be a numpy array'):
        traj_data['env_infos']['bar'] = dict()
        t = TrajectoryBatch(**traj_data)
        del t


def test_env_infos_batch_mismatch(traj_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must have a batch dimension'):
        traj_data['env_infos']['goal'] = traj_data['env_infos']['goal'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_agent_infos_not_ndarray(traj_data):
    with pytest.raises(ValueError,
                       match='entry in agent_infos must be a numpy array'):
        traj_data['agent_infos']['bar'] = list()
        t = TrajectoryBatch(**traj_data)
        del t


def test_agent_infos_batch_mismatch(traj_data):
    with pytest.raises(
            ValueError,
            match='entry in agent_infos must have a batch dimension'):
        traj_data['agent_infos']['hidden'] = traj_data['agent_infos'][
            'hidden'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t

import akro
import gym.spaces
import numpy as np
import pytest

from garage import TimeStep
from garage import TimeStepBatch
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
    last_obs = np.stack([obs_space.low] * len(lens))
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
        'last_observations': last_obs,
        'actions': act,
        'rewards': rew,
        'terminals': terms,
        'env_infos': env_infos,
        'agent_infos': agent_infos,
        'lengths': lens,
    }


def test_new_traj(traj_data):
    t = TrajectoryBatch(**traj_data)
    assert t.env_spec is traj_data['env_spec']
    assert t.observations is traj_data['observations']
    assert t.last_observations is traj_data['last_observations']
    assert t.actions is traj_data['actions']
    assert t.rewards is traj_data['rewards']
    assert t.terminals is traj_data['terminals']
    assert t.env_infos is traj_data['env_infos']
    assert t.agent_infos is traj_data['agent_infos']
    assert t.lengths is traj_data['lengths']


def test_lengths_shape_mismatch_traj(traj_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must be a tensor of shape'):
        traj_data['lengths'] = traj_data['lengths'].reshape((4, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_lengths_dtype_mismatch_traj(traj_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must have an integer dtype'):
        traj_data['lengths'] = traj_data['lengths'].astype(np.float32)
        t = TrajectoryBatch(**traj_data)
        del t


def test_obs_env_spec_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='observations must conform'):
        traj_data['observations'] = traj_data['observations'][:, :, :, :1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_obs_batch_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='batch dimension of observations'):
        traj_data['observations'] = traj_data['observations'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_last_obs_env_spec_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='last_observations must conform'):
        traj_data['last_observations'] = \
                traj_data['last_observations'][:, :, :, :1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_last_obs_batch_mismatch_traj(traj_data):
    with pytest.raises(ValueError,
                       match='batch dimension of last_observations'):
        traj_data['last_observations'] = traj_data['last_observations'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_act_env_spec_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='actions must conform'):
        traj_data['actions'] = traj_data['actions'][:, 0]
        t = TrajectoryBatch(**traj_data)
        del t


def test_act_box_env_spec_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='actions should have'):
        traj_data['env_spec'].action_space = akro.Box(low=1,
                                                      high=np.inf,
                                                      shape=(4, 3, 2),
                                                      dtype=np.float32)
        t = TrajectoryBatch(**traj_data)
        del t


def test_act_batch_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='batch dimension of actions'):
        traj_data['actions'] = traj_data['actions'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_rewards_shape_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='Rewards tensor'):
        traj_data['rewards'] = traj_data['rewards'].reshape((2, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_terminals_shape_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='terminals tensor must have shape'):
        traj_data['terminals'] = traj_data['terminals'].reshape((2, -1))
        t = TrajectoryBatch(**traj_data)
        del t


def test_terminals_dtype_mismatch_traj(traj_data):
    with pytest.raises(ValueError, match='terminals tensor must be dtype'):
        traj_data['terminals'] = traj_data['terminals'].astype(np.float32)
        t = TrajectoryBatch(**traj_data)
        del t


def test_env_infos_not_ndarray_traj(traj_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must be a numpy array'):
        traj_data['env_infos']['bar'] = []
        t = TrajectoryBatch(**traj_data)
        del t


def test_env_infos_batch_mismatch_traj(traj_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must have a batch dimension'):
        traj_data['env_infos']['goal'] = traj_data['env_infos']['goal'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_agent_infos_not_ndarray_traj(traj_data):
    with pytest.raises(ValueError,
                       match='entry in agent_infos must be a numpy array'):
        traj_data['agent_infos']['bar'] = list()
        t = TrajectoryBatch(**traj_data)
        del t


def test_agent_infos_batch_mismatch_traj(traj_data):
    with pytest.raises(
            ValueError,
            match='entry in agent_infos must have a batch dimension'):
        traj_data['agent_infos']['hidden'] = traj_data['agent_infos'][
            'hidden'][:-1]
        t = TrajectoryBatch(**traj_data)
        del t


def test_to_trajectory_list(traj_data):
    t = TrajectoryBatch(**traj_data)
    t_list = t.to_trajectory_list()
    assert len(t_list) == len(traj_data['lengths'])
    start = 0
    for length, last_obs, s in zip(traj_data['lengths'],
                                   traj_data['last_observations'], t_list):
        stop = start + length
        assert (
            s['observations'] == traj_data['observations'][start:stop]).all()
        assert (s['next_observations'] == np.concatenate(
            (traj_data['observations'][start + 1:stop], [last_obs]))).all()
        assert (s['actions'] == traj_data['actions'][start:stop]).all()
        assert (s['rewards'] == traj_data['rewards'][start:stop]).all()
        assert (s['dones'] == traj_data['terminals'][start:stop]).all()
        start = stop
    assert start == len(traj_data['rewards'])


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
    obs = obs_space.sample()
    next_obs = obs_space.sample()
    act = act_space.sample()
    rew = 10.0
    terms = False

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.array([[1, 1]])
    env_infos['TimeLimit.truncated'] = not terms

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act

    return {
        'env_spec': env_spec,
        'observation': obs,
        'next_observation': next_obs,
        'action': act,
        'reward': rew,
        'terminal': terms,
        'env_info': env_infos,
        'agent_info': agent_infos,
    }


def test_new_time_step(sample_data):
    s = TimeStep(**sample_data)
    assert s.env_spec is sample_data['env_spec']
    assert s.observation is sample_data['observation']
    assert s.action is sample_data['action']
    assert s.reward is sample_data['reward']
    assert s.terminal is sample_data['terminal']
    assert s.env_info is sample_data['env_info']
    assert s.agent_info is sample_data['agent_info']
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
    sample_data['next_observation'] = obs_space.sample()
    sample_data['action'] = act_space.sample()
    s = TimeStep(**sample_data)

    assert s.observation is sample_data['observation']
    assert s.next_observation is sample_data['next_observation']
    assert s.action is sample_data['action']


def test_obs_env_spec_mismatch_time_step(sample_data):
    with pytest.raises(ValueError,
                       match='observation must conform to observation_space'):
        sample_data['observation'] = sample_data['observation'][:, :, :1]
        s = TimeStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 5, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='observation should have the same dimensionality'):
        sample_data['observation'] = sample_data['observation'][:, :, :1]
        s = TimeStep(**sample_data)
        del s


def test_next_obs_env_spec_mismatch_time_step(sample_data):
    with pytest.raises(
            ValueError,
            match='next_observation must conform to observation_space'):
        sample_data['next_observation'] = sample_data[
            'next_observation'][:, :, :1]
        s = TimeStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='next_observation should have the same dimensionality'):
        sample_data['next_observation'] = sample_data[
            'next_observation'][:, :, :1]
        s = TimeStep(**sample_data)
        del s


def test_act_env_spec_mismatch_time_step(sample_data):
    with pytest.raises(ValueError,
                       match='action must conform to action_space'):
        sample_data['action'] = sample_data['action'][:-1]
        s = TimeStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = akro.Discrete(5)
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(ValueError,
                       match='action should have the same dimensionality'):
        sample_data['action'] = sample_data['action'][:-1]
        s = TimeStep(**sample_data)
        del s


def test_reward_dtype_mismatch_time_step(sample_data):
    with pytest.raises(ValueError, match='reward must be type'):
        sample_data['reward'] = []
        s = TimeStep(**sample_data)
        del s


def test_terminal_dtype_mismatch_time_step(sample_data):
    with pytest.raises(ValueError, match='terminal must be dtype bool'):
        sample_data['terminal'] = []
        s = TimeStep(**sample_data)
        del s


def test_agent_info_dtype_mismatch_time_step(sample_data):
    with pytest.raises(ValueError, match='agent_info must be type'):
        sample_data['agent_info'] = []
        s = TimeStep(**sample_data)
        del s


def test_env_info_dtype_mismatch_time_step(sample_data):
    with pytest.raises(ValueError, match='env_info must be type'):
        sample_data['env_info'] = []
        s = TimeStep(**sample_data)
        del s


@pytest.fixture
def batch_data():
    # spaces
    obs_space = gym.spaces.Box(low=1,
                               high=np.inf,
                               shape=(4, 3, 2),
                               dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)

    # generate data
    batch_size = 2
    obs = np.stack([obs_space.low] * batch_size)
    next_obs = np.stack([obs_space.low] * batch_size)
    act = np.stack([[1, 3]] * batch_size)
    rew = np.arange(batch_size)
    terms = np.zeros(batch_size, dtype=np.bool)
    terms[np.cumsum(batch_size) - 1] = True  # set terminal bits

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.stack([[1, 1]] * batch_size)
    env_infos['foo'] = np.arange(batch_size)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act
    agent_infos['hidden'] = np.arange(batch_size)

    return {
        'env_spec': env_spec,
        'observations': obs,
        'next_observations': next_obs,
        'actions': act,
        'rewards': rew,
        'terminals': terms,
        'env_infos': env_infos,
        'agent_infos': agent_infos,
    }


def test_new_ts_batch(batch_data):
    s = TimeStepBatch(**batch_data)
    assert s.env_spec is batch_data['env_spec']
    assert s.observations is batch_data['observations']
    assert s.next_observations is batch_data['next_observations']
    assert s.actions is batch_data['actions']
    assert s.rewards is batch_data['rewards']
    assert s.terminals is batch_data['terminals']
    assert s.env_infos is batch_data['env_infos']
    assert s.agent_infos is batch_data['agent_infos']


def test_observations_env_spec_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='observations must conform'):
        batch_data['observations'] = batch_data['observations'][:, :, :, :1]
        s = TimeStepBatch(**batch_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 5, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    batch_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='observations should have the same dimensionality'):
        batch_data['observations'] = batch_data['observations'][:, :, :, :1]
        s = TimeStepBatch(**batch_data)
        del s


def test_observations_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='batch dimension of observations'):
        batch_data['observations'] = batch_data['observations'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_next_observations_env_spec_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='next_observations must conform'):
        batch_data['next_observations'] = batch_data[
            'next_observations'][:, :, :, :1]
        s = TimeStepBatch(**batch_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    batch_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='next_observations should have the same dimensionality'):
        batch_data['next_observations'] = batch_data[
            'next_observations'][:, :, :, :1]
        s = TimeStepBatch(**batch_data)
        del s


def test_next_observations_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError,
                       match='batch dimension of '
                       'next_observations'):
        batch_data['next_observations'] = batch_data['next_observations'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_actions_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='batch dimension of actions'):
        batch_data['actions'] = batch_data['actions'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_rewards_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='batch dimension of rewards'):
        batch_data['rewards'] = batch_data['rewards'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_act_env_spec_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='actions must conform'):
        batch_data['actions'] = batch_data['actions'][:, 0]
        s = TimeStepBatch(**batch_data)
        del s


def test_act_box_env_spec_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='actions should have'):
        batch_data['env_spec'].action_space = akro.Box(low=1,
                                                       high=np.inf,
                                                       shape=(4, 3, 2),
                                                       dtype=np.float32)
        s = TimeStepBatch(**batch_data)
        del s


def test_empty_terminals__batch(batch_data):
    with pytest.raises(ValueError, match='batch dimension of terminals'):
        batch_data['terminals'] = []
        s = TimeStepBatch(**batch_data)
        del s


def test_terminals_dtype_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='terminals tensor must be dtype'):
        batch_data['terminals'] = batch_data['terminals'].astype(np.float32)
        s = TimeStepBatch(**batch_data)
        del s


def test_env_infos_not_ndarray_batch(batch_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must be a numpy array'):
        batch_data['env_infos']['bar'] = []
        s = TimeStepBatch(**batch_data)
        del s


def test_env_infos_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must have a batch dimension'):
        batch_data['env_infos']['goal'] = batch_data['env_infos']['goal'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_agent_infos_not_ndarray_batch(batch_data):
    with pytest.raises(ValueError,
                       match='entry in agent_infos must be a numpy array'):
        batch_data['agent_infos']['bar'] = list()
        s = TimeStepBatch(**batch_data)
        del s


def test_agent_infos_batch_mismatch_batch(batch_data):
    with pytest.raises(
            ValueError,
            match='entry in agent_infos must have a batch dimension'):
        batch_data['agent_infos']['hidden'] = batch_data['agent_infos'][
            'hidden'][:-1]
        s = TimeStepBatch(**batch_data)
        del s


def test_concatenate_batch(batch_data):
    single_batch = TimeStepBatch(**batch_data)
    batches = [single_batch, single_batch]
    s = TimeStepBatch.concatenate(*batches)

    new_obs = np.concatenate(
        [batch_data['observations'], batch_data['observations']])
    new_next_obs = np.concatenate(
        [batch_data['next_observations'], batch_data['next_observations']])
    new_actions = np.concatenate(
        [batch_data['actions'], batch_data['actions']])
    new_rewards = np.concatenate(
        [batch_data['rewards'], batch_data['rewards']])
    new_terminals = np.concatenate(
        [batch_data['terminals'], batch_data['terminals']])
    new_env_infos = {
        k: np.concatenate([b.env_infos[k] for b in batches])
        for k in batches[0].env_infos.keys()
    }
    new_agent_infos = {
        k: np.concatenate([b.agent_infos[k] for b in batches])
        for k in batches[0].agent_infos.keys()
    }

    assert s.env_spec == batch_data['env_spec']
    assert np.array_equal(s.observations, new_obs)
    assert np.array_equal(s.next_observations, new_next_obs)
    assert np.array_equal(s.actions, new_actions)
    assert np.array_equal(s.rewards, new_rewards)
    assert np.array_equal(s.terminals, new_terminals)
    for key in new_env_infos:
        assert key in s.env_infos
        assert np.array_equal(new_env_infos[key], s.env_infos[key])
    for key in new_agent_infos:
        assert key in s.agent_infos
        assert np.array_equal(new_agent_infos[key], s.agent_infos[key])


def test_concatenate_empty_batch():
    with pytest.raises(ValueError, match='at least one'):
        batches = []
        s = TimeStepBatch.concatenate(*batches)
        del s


def test_split_batch(batch_data):
    s = TimeStepBatch(
        env_spec=batch_data['env_spec'],
        observations=batch_data['observations'],
        actions=batch_data['actions'],
        rewards=batch_data['rewards'],
        next_observations=batch_data['next_observations'],
        terminals=batch_data['terminals'],
        env_infos=batch_data['env_infos'],
        agent_infos=batch_data['agent_infos'],
    )
    batches = s.split()

    assert len(batches) == 2  # original batch_data is a batch of 2
    for i, batch in enumerate(batches):
        assert batch.env_spec == batch_data['env_spec']
        assert np.array_equal(batch.observations,
                              [batch_data['observations'][i]])
        assert np.array_equal(batch.next_observations,
                              [batch_data['next_observations'][i]])
        assert np.array_equal(batch.actions, [batch_data['actions'][i]])
        assert np.array_equal(batch.rewards, [batch_data['rewards'][i]])
        assert np.array_equal(batch.terminals, [batch_data['terminals'][i]])
        for key in batch.env_infos:
            assert key in batch_data['env_infos']
            assert np.array_equal(batch.env_infos[key],
                                  [batch_data['env_infos'][key][i]])
        for key in batch.agent_infos:
            assert key in batch_data['agent_infos']
            assert (np.array_equal(batch.agent_infos[key],
                                   [batch_data['agent_infos'][key][i]]))


def test_to_time_step_list_batch(batch_data):
    s = TimeStepBatch(
        env_spec=batch_data['env_spec'],
        observations=batch_data['observations'],
        actions=batch_data['actions'],
        rewards=batch_data['rewards'],
        next_observations=batch_data['next_observations'],
        terminals=batch_data['terminals'],
        env_infos=batch_data['env_infos'],
        agent_infos=batch_data['agent_infos'],
    )
    batches = s.to_time_step_list()

    assert len(batches) == 2  # original batch_data is a batch of 2
    for i, batch in enumerate(batches):
        assert np.array_equal(batch['observations'],
                              [batch_data['observations'][i]])
        assert np.array_equal(batch['next_observations'],
                              [batch_data['next_observations'][i]])
        assert np.array_equal(batch['actions'], [batch_data['actions'][i]])
        assert np.array_equal(batch['rewards'], [batch_data['rewards'][i]])
        assert np.array_equal(batch['terminals'], [batch_data['terminals'][i]])
        for key in batch['env_infos']:
            assert key in batch_data['env_infos']
            assert np.array_equal(batch['env_infos'][key],
                                  [batch_data['env_infos'][key][i]])
        for key in batch['agent_infos']:
            assert key in batch_data['agent_infos']
            assert np.array_equal(batch['agent_infos'][key],
                                  [batch_data['agent_infos'][key][i]])


def test_from_empty_time_step_list_batch(batch_data):
    with pytest.raises(ValueError, match='at least one dict'):
        batches = []
        s = TimeStepBatch.from_time_step_list(batch_data['env_spec'], batches)
        del s


def test_from_time_step_list_batch(batch_data):
    batches = [batch_data, batch_data]
    s = TimeStepBatch.from_time_step_list(batch_data['env_spec'], batches)

    new_obs = np.concatenate(
        [batch_data['observations'], batch_data['observations']])
    new_next_obs = np.concatenate(
        [batch_data['next_observations'], batch_data['next_observations']])
    new_actions = np.concatenate(
        [batch_data['actions'], batch_data['actions']])
    new_rewards = np.concatenate(
        [batch_data['rewards'], batch_data['rewards']])
    new_terminals = np.concatenate(
        [batch_data['terminals'], batch_data['terminals']])
    new_env_infos = {
        k: np.concatenate([b['env_infos'][k] for b in batches])
        for k in batches[0]['env_infos'].keys()
    }
    new_agent_infos = {
        k: np.concatenate([b['agent_infos'][k] for b in batches])
        for k in batches[0]['agent_infos'].keys()
    }

    assert s.env_spec == batch_data['env_spec']
    assert np.array_equal(s.observations, new_obs)
    assert np.array_equal(s.next_observations, new_next_obs)
    assert np.array_equal(s.actions, new_actions)
    assert np.array_equal(s.rewards, new_rewards)
    assert np.array_equal(s.terminals, new_terminals)
    for key in new_env_infos:
        assert key in s.env_infos
        assert np.array_equal(new_env_infos[key], s.env_infos[key])
    for key in new_agent_infos:
        assert key in s.agent_infos
        assert np.array_equal(new_agent_infos[key], s.agent_infos[key])

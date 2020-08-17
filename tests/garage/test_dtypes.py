import akro
import gym.spaces
import numpy as np
import pytest

# yapf: disable
from garage import (EnvSpec,
                    EnvStep,
                    EpisodeBatch,
                    StepType,
                    TimeStep,
                    TimeStepBatch)

# yapf: enable


@pytest.fixture
def eps_data():
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

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.stack([[1, 1]] * n_t)
    env_infos['foo'] = np.arange(n_t)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act
    agent_infos['hidden'] = np.arange(n_t)

    # step_types
    step_types = []
    for size in lens:
        step_types.extend([StepType.FIRST] + [StepType.MID] * (size - 2) +
                          [StepType.TERMINAL])
    step_types = np.array(step_types, dtype=StepType)

    return {
        'env_spec': env_spec,
        'observations': obs,
        'last_observations': last_obs,
        'actions': act,
        'rewards': rew,
        'env_infos': env_infos,
        'agent_infos': agent_infos,
        'step_types': step_types,
        'lengths': lens
    }


def test_new_eps(eps_data):
    t = EpisodeBatch(**eps_data)
    assert t.env_spec is eps_data['env_spec']
    assert t.observations is eps_data['observations']
    assert t.last_observations is eps_data['last_observations']
    assert t.actions is eps_data['actions']
    assert t.rewards is eps_data['rewards']
    assert t.env_infos is eps_data['env_infos']
    assert t.agent_infos is eps_data['agent_infos']
    assert t.step_types is eps_data['step_types']
    assert t.lengths is eps_data['lengths']


def test_lengths_shape_mismatch_eps(eps_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must be a tensor of shape'):
        eps_data['lengths'] = eps_data['lengths'].reshape((4, -1))
        t = EpisodeBatch(**eps_data)
        del t


def test_lengths_dtype_mismatch_eps(eps_data):
    with pytest.raises(ValueError,
                       match='Lengths tensor must have an integer dtype'):
        eps_data['lengths'] = eps_data['lengths'].astype(np.float32)
        t = EpisodeBatch(**eps_data)
        del t


def test_obs_env_spec_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='observations must conform'):
        eps_data['observations'] = eps_data['observations'][:, :, :, :1]
        t = EpisodeBatch(**eps_data)
        del t


def test_obs_batch_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='batch dimension of observations'):
        eps_data['observations'] = eps_data['observations'][:-1]
        t = EpisodeBatch(**eps_data)
        del t


def test_last_obs_env_spec_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='last_observations must conform'):
        eps_data['last_observations'] = \
                eps_data['last_observations'][:, :, :, :1]
        t = EpisodeBatch(**eps_data)
        del t


def test_last_obs_batch_mismatch_eps(eps_data):
    with pytest.raises(ValueError,
                       match='batch dimension of last_observations'):
        eps_data['last_observations'] = eps_data['last_observations'][:-1]
        t = EpisodeBatch(**eps_data)
        del t


def test_act_env_spec_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='actions must conform'):
        eps_data['actions'] = eps_data['actions'][:, 0]
        t = EpisodeBatch(**eps_data)
        del t


def test_act_box_env_spec_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='actions should have'):
        eps_data['env_spec'].action_space = akro.Box(low=1,
                                                     high=np.inf,
                                                     shape=(4, 3, 2),
                                                     dtype=np.float32)
        t = EpisodeBatch(**eps_data)
        del t


def test_act_batch_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='batch dimension of actions'):
        eps_data['actions'] = eps_data['actions'][:-1]
        t = EpisodeBatch(**eps_data)
        del t


def test_rewards_shape_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='Rewards tensor'):
        eps_data['rewards'] = eps_data['rewards'].reshape((2, -1))
        t = EpisodeBatch(**eps_data)
        del t


def test_env_infos_not_ndarray_eps(eps_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must be a numpy array'):
        eps_data['env_infos']['bar'] = []
        t = EpisodeBatch(**eps_data)
        del t


def test_env_infos_batch_mismatch_eps(eps_data):
    with pytest.raises(ValueError,
                       match='entry in env_infos must have a batch dimension'):
        eps_data['env_infos']['goal'] = eps_data['env_infos']['goal'][:-1]
        t = EpisodeBatch(**eps_data)
        del t


def test_agent_infos_not_ndarray_eps(eps_data):
    with pytest.raises(ValueError,
                       match='entry in agent_infos must be a numpy array'):
        eps_data['agent_infos']['bar'] = list()
        t = EpisodeBatch(**eps_data)
        del t


def test_agent_infos_batch_mismatch_eps(eps_data):
    with pytest.raises(
            ValueError,
            match='entry in agent_infos must have a batch dimension'):
        eps_data['agent_infos']['hidden'] = eps_data['agent_infos'][
            'hidden'][:-1]
        t = EpisodeBatch(**eps_data)
        del t


def test_step_types_shape_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='step_types tensor must have shape'):
        eps_data['step_types'] = eps_data['step_types'].reshape((2, -1))
        t = EpisodeBatch(**eps_data)
        del t


def test_step_types_dtype_mismatch_eps(eps_data):
    with pytest.raises(ValueError, match='step_types tensor must be dtype'):
        eps_data['step_types'] = eps_data['step_types'].astype(np.float32)
        t = EpisodeBatch(**eps_data)
        del t


def test_to_epsectory_list(eps_data):
    t = EpisodeBatch(**eps_data)
    t_list = t.to_list()
    assert len(t_list) == len(eps_data['lengths'])
    start = 0
    for length, last_obs, s in zip(eps_data['lengths'],
                                   eps_data['last_observations'], t_list):
        stop = start + length
        assert (
            s['observations'] == eps_data['observations'][start:stop]).all()
        assert (s['next_observations'] == np.concatenate(
            (eps_data['observations'][start + 1:stop], [last_obs]))).all()
        assert (s['actions'] == eps_data['actions'][start:stop]).all()
        assert (s['rewards'] == eps_data['rewards'][start:stop]).all()
        assert (s['step_types'] == eps_data['step_types'][start:stop]).all()
        start = stop
    assert start == len(eps_data['rewards'])


def test_get_step_type():
    step_type = StepType.get_step_type(step_cnt=1,
                                       max_episode_length=5,
                                       done=False)
    assert step_type == StepType.FIRST
    step_type = StepType.get_step_type(step_cnt=2,
                                       max_episode_length=5,
                                       done=False)
    assert step_type == StepType.MID
    step_type = StepType.get_step_type(step_cnt=2,
                                       max_episode_length=None,
                                       done=False)
    assert step_type == StepType.MID
    step_type = StepType.get_step_type(step_cnt=5,
                                       max_episode_length=5,
                                       done=False)
    assert step_type == StepType.TIMEOUT
    step_type = StepType.get_step_type(step_cnt=5,
                                       max_episode_length=5,
                                       done=True)
    assert step_type == StepType.TIMEOUT
    step_type = StepType.get_step_type(step_cnt=1,
                                       max_episode_length=5,
                                       done=True)
    assert step_type == StepType.TERMINAL
    with pytest.raises(ValueError):
        step_type = StepType.get_step_type(step_cnt=0,
                                           max_episode_length=5,
                                           done=False)


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
    step_type = StepType.FIRST

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.array([[1, 1]])
    env_infos['TimeLimit.truncated'] = (step_type == StepType.TIMEOUT)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act

    return {
        'env_spec': env_spec,
        'observation': obs,
        'next_observation': next_obs,
        'action': act,
        'reward': rew,
        'env_info': env_infos,
        'agent_info': agent_infos,
        'step_type': step_type
    }


def test_new_time_step(sample_data):
    s = TimeStep(**sample_data)
    assert s.env_spec is sample_data['env_spec']
    assert s.observation is sample_data['observation']
    assert s.action is sample_data['action']
    assert s.reward is sample_data['reward']
    assert s.step_type is sample_data['step_type']
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


def test_step_type_property_time_step(sample_data):
    sample_data['step_type'] = StepType.FIRST
    s = TimeStep(**sample_data)
    assert s.first

    sample_data['step_type'] = StepType.MID
    s = TimeStep(**sample_data)
    assert s.mid

    sample_data['step_type'] = StepType.TERMINAL
    s = TimeStep(**sample_data)
    assert s.terminal and s.last

    sample_data['step_type'] = StepType.TIMEOUT
    s = TimeStep(**sample_data)
    assert s.timeout and s.last


def test_from_env_step_time_step(sample_data):
    agent_info = sample_data['agent_info']
    last_observation = sample_data['observation']
    observation = sample_data['next_observation']
    time_step = TimeStep(**sample_data)
    del sample_data['agent_info']
    del sample_data['next_observation']
    sample_data['observation'] = observation
    env_step = EnvStep(**sample_data)
    time_step_new = TimeStep.from_env_step(env_step=env_step,
                                           last_observation=last_observation,
                                           agent_info=agent_info)
    assert time_step == time_step_new


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
    step_types = np.array([StepType.FIRST, StepType.TERMINAL], dtype=StepType)

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
        'env_infos': env_infos,
        'agent_infos': agent_infos,
        'step_types': step_types
    }


def test_new_ts_batch(batch_data):
    s = TimeStepBatch(**batch_data)
    assert s.env_spec is batch_data['env_spec']
    assert s.observations is batch_data['observations']
    assert s.next_observations is batch_data['next_observations']
    assert s.actions is batch_data['actions']
    assert s.rewards is batch_data['rewards']
    assert s.env_infos is batch_data['env_infos']
    assert s.agent_infos is batch_data['agent_infos']
    assert s.step_types is batch_data['step_types']


def test_invalid_inferred_batch_size(batch_data):
    with pytest.raises(ValueError, match='batch dimension of rewards'):
        batch_data['rewards'] = []
        s = TimeStepBatch(**batch_data)
        del s


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


def test_step_types_batch_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='batch dimension of step_types'):
        batch_data['step_types'] = np.array([])
        s = TimeStepBatch(**batch_data)
        del s


def test_step_types_dtype_mismatch_batch(batch_data):
    with pytest.raises(ValueError, match='step_types must be a StepType'):
        batch_data['step_types'] = batch_data['step_types'].astype(np.float32)
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
    new_step_types = np.concatenate(
        [batch_data['step_types'], batch_data['step_types']])
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
    assert np.array_equal(s.step_types, new_step_types)
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
        step_types=batch_data['step_types'],
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
        assert np.array_equal(batch.step_types, [batch_data['step_types'][i]])
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
        step_types=batch_data['step_types'],
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
        assert np.array_equal(batch['step_types'],
                              [batch_data['step_types'][i]])
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
    new_step_types = np.concatenate(
        [batch_data['step_types'], batch_data['step_types']])
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
    assert np.array_equal(s.step_types, new_step_types)
    for key in new_env_infos:
        assert key in s.env_infos
        assert np.array_equal(new_env_infos[key], s.env_infos[key])
    for key in new_agent_infos:
        assert key in s.agent_infos
        assert np.array_equal(new_agent_infos[key], s.agent_infos[key])


def test_time_step_batch_from_episode_batch(eps_data):
    eps = EpisodeBatch(**eps_data)
    timestep_batch = TimeStepBatch.from_episode_batch(eps)
    assert (timestep_batch.observations == eps.observations).all()
    assert (timestep_batch.next_observations[:eps.lengths[0] - 1] ==
            eps.observations[1:eps.lengths[0]]).all()
    assert (timestep_batch.next_observations[eps.lengths[0]] ==
            eps.last_observations[0]).all()

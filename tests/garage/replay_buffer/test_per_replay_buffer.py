import akro
import numpy as np
import pytest

from garage import EnvSpec, EpisodeBatch, StepType
from garage.replay_buffer import PERReplayBuffer

from tests.fixtures.envs.dummy import DummyDiscreteEnv


@pytest.fixture
def setup():
    obs_space = akro.Box(low=1, high=np.inf, shape=(1, ), dtype=np.float32)
    act_space = akro.Discrete(1)
    env_spec = EnvSpec(obs_space, act_space)
    buffer = PERReplayBuffer(100, 100, env_spec)
    return buffer, DummyDiscreteEnv()


def test_add_path(setup):
    buff, env = setup
    obs = env.reset()
    buff.add_path({'observations': np.array([obs for _ in range(5)])})

    # initial priorities for inserted timesteps should be 1
    assert (buff._priorities[:5] == 1.).all()
    assert (buff._priorities[5:] == 0.).all()

    # test case where buffer is full and paths are split
    # into two segments
    num_obs = buff._capacity - buff._transitions_stored
    buff.add_path(
        {'observations': np.array([obs for _ in range(num_obs - 1)])})

    # artificially set the priority of a transition to be high .
    # the next path added to the buffer should wrap around the buffer
    # and contain one timestep at the end and 5 at the beginning, all
    # of which should have priority == max(buff._priorities).
    buff._priorities[-1] = 100.
    buff.add_path({'observations': np.array([obs for _ in range(6)])})

    assert buff._priorities[-1] == 100.
    assert (buff._priorities[:5] == 100.).all()


def test_update_priorities(setup):
    buff, env = setup
    obs = env.reset()
    buff.add_path({'observations': np.array([obs for _ in range(5)])})

    assert (buff._priorities[:5] == 1.).all()
    assert (buff._priorities[5:] == 0.).all()

    indices = list(range(2, 10))
    new_priorities = [0.5 for _ in range(2, 10)]
    buff.update_priorities(indices, new_priorities)

    assert (buff._priorities[2:10] == 0.5).all()
    assert (buff._priorities[:2] != 0.5).all()
    assert (buff._priorities[10:] != 0.5).all()


@pytest.mark.parametrize('alpha, beta_init', [(0.5, 0.5), (0.4, 0.6),
                                              (0.1, 0.9)])
def test_sample_transitions(setup, alpha, beta_init):
    buff, env = setup
    obs = env.reset()
    buff.add_path({
        'observations':
        np.array([np.full_like(obs, i, dtype=np.float32) for i in range(50)]),
    })

    buff._beta_init = beta_init
    buff._alpha = alpha
    transitions, weights, indices = buff.sample_transitions(50)
    obses = transitions['observations']

    # verify the indices returned correspond to the correct samples
    for o, i in zip(obses, indices):
        assert (o == i).all()

    # verify the weights are correct
    probs = buff._priorities**buff._alpha
    probs /= probs.sum()

    beta = buff._beta_init + 50 * (1.0 - buff._beta_init) / 100
    beta = min(1.0, beta)
    w = (50 * probs[indices])**(-beta)
    w /= w.max()
    w = np.array(w)

    assert (w == weights).all()


def test_sample_timesteps(setup):
    buff, env = setup
    obs = env.reset()
    buff.add_path({
        'observations':
        np.array([np.full_like(obs, i, dtype=np.float32) for i in range(50)]),
        'next_observations':
        np.array([np.full_like(obs, i, dtype=np.float32) for i in range(50)]),
        'actions':
        np.array([np.full_like(obs, i, dtype=np.float32) for i in range(50)]),
        'terminals':
        np.array([[False] for _ in range(50)]),
        'rewards':
        np.array([[1] for _ in range(50)])
    })

    timesteps, weights, indices = buff.sample_timesteps(50)

    assert len(weights) == 50
    assert len(indices) == 50

    obses, actions = timesteps.observations, timesteps.actions

    for a, o, i in zip(actions, obses, indices):
        assert (o == i).all()
        assert (a == i).all()

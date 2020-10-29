import pickle

import numpy as np
import pytest
import torch

from garage.envs import GymEnv
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction

from tests.fixtures.envs.dummy import DummyBoxEnv


@pytest.mark.parametrize('batch_size', [1, 5, 10])
def test_forward(batch_size):
    env_spec = GymEnv(DummyBoxEnv()).spec
    obs_dim = env_spec.observation_space.flat_dim
    obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
    qf = DiscreteMLPQFunction(env_spec=env_spec,
                              hidden_nonlinearity=None,
                              hidden_sizes=(2, 2))
    qvals = qf(obs)
    policy = DiscreteQFArgmaxPolicy(qf, env_spec)
    assert (policy(obs) == torch.argmax(qvals, dim=1)).all()
    assert policy(obs).shape == (batch_size, )


def test_get_action():
    env_spec = GymEnv(DummyBoxEnv()).spec
    obs_dim = env_spec.observation_space.flat_dim
    obs = torch.ones([
        obs_dim,
    ], dtype=torch.float32)
    qf = DiscreteMLPQFunction(env_spec=env_spec,
                              hidden_nonlinearity=None,
                              hidden_sizes=(2, 2))
    qvals = qf(obs.unsqueeze(0))
    policy = DiscreteQFArgmaxPolicy(qf, env_spec)
    action, _ = policy.get_action(obs.numpy())
    assert action == torch.argmax(qvals, dim=1).numpy()
    assert action.shape == ()


@pytest.mark.parametrize('batch_size', [1, 5, 10])
def test_get_actions(batch_size):
    env_spec = GymEnv(DummyBoxEnv()).spec
    obs_dim = env_spec.observation_space.flat_dim
    obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
    qf = DiscreteMLPQFunction(env_spec=env_spec,
                              hidden_nonlinearity=None,
                              hidden_sizes=(2, 2))
    qvals = qf(obs)
    policy = DiscreteQFArgmaxPolicy(qf, env_spec)
    actions, _ = policy.get_actions(obs.numpy())
    assert (actions == torch.argmax(qvals, dim=1).numpy()).all()
    assert actions.shape == (batch_size, )


@pytest.mark.parametrize('batch_size', [1, 5, 10])
def test_is_pickleable(batch_size):
    env_spec = GymEnv(DummyBoxEnv())
    obs_dim = env_spec.observation_space.flat_dim
    obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
    qf = DiscreteMLPQFunction(env_spec=env_spec,
                              hidden_nonlinearity=None,
                              hidden_sizes=(2, 2))
    policy = DiscreteQFArgmaxPolicy(qf, env_spec)

    output1 = policy.get_actions(obs.numpy())[0]

    p = pickle.dumps(policy)
    policy_pickled = pickle.loads(p)
    output2 = policy_pickled.get_actions(obs.numpy())[0]
    assert np.array_equal(output1, output2)

import pickle

import numpy as np
import pytest

from garage.envs import GymEnv
from garage.torch.policies import GaussianLSTMPolicy

# yapf: disable
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDictEnv

# yapf: enable


def test_get_action_dict_space():
    env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
    policy = GaussianLSTMPolicy(env_spec=env.spec, hidden_size=4)
    policy.reset(do_resets=None)
    obs = env.reset()[0]

    action, _ = policy.get_action(obs)
    assert env.action_space.contains(action)

    policy.reset(do_resets=[True, True])

    actions, _ = policy.get_actions([obs, obs])
    for action in actions:
        assert env.action_space.contains(action)


# yapf: disable
@pytest.mark.parametrize('obs_dim, action_dim, hidden_size', [
    ((1, ), (1, ), 4),
    ((2, ), (2, ), 4),
    ((1, 1), (1, ), 4),
    ((2, 2), (2, ), 4)
])
# yapf: enable
def test_get_action(obs_dim, action_dim, hidden_size):
    env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
    policy = GaussianLSTMPolicy(env_spec=env.spec, hidden_size=hidden_size)

    policy.reset()
    obs = env.reset()[0]

    action, _ = policy.get_action(obs.flatten())
    assert env.action_space.contains(action)

    actions, _ = policy.get_actions([obs.flatten()])
    for action in actions:
        assert env.action_space.contains(action)


# pylint: disable=no-member
def test_is_pickleable():
    env = GymEnv(DummyBoxEnv(obs_dim=(1, ), action_dim=(1, )))
    policy = GaussianLSTMPolicy(env_spec=env.spec)
    policy.reset()
    obs = env.reset()[0]
    _, info = policy.get_action(obs)

    p = pickle.dumps(policy)
    policy_pickled = pickle.loads(p)
    policy_pickled.reset()
    _, info2 = policy_pickled.get_action(obs)
    assert np.array_equal(info['mean'], info2['mean'])
    assert np.array_equal(info['log_std'], info2['log_std'])

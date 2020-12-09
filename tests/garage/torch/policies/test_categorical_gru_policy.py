import cloudpickle

import numpy as np
import pytest
import torch

from garage.envs import GymEnv
from garage.torch.policies import CategoricalGRUPolicy

# yapf: disable
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestCategoricalGRUPolicy:

    def test_invalid_env(self):
        env = GymEnv(DummyBoxEnv())
        with pytest.raises(ValueError):
            CategoricalGRUPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=True)
        policy.reset()
        obs = env.reset()[0]
        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim, obs_type', [
        ((1, ), 1, 4, 'discrete'),
        ((2, ), 2, 4, 'discrete'),
        ((1, 1), 1, 4, 'discrete'),
        ((2, 2), 2, 4, 'discrete'),
        ((1, ), 1, 4, 'dict'),
    ])
    def test_get_action(self, obs_dim, action_dim, hidden_dim, obs_type):
        assert obs_type in ['discrete', 'dict']
        if obs_type == 'discrete':
            env = GymEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GymEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()[0]

        if obs_type == 'discrete':
            obs = obs.flatten()

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs])
        for action in actions:
            assert env.action_space.contains(action)


    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim, obs_type', [
        ((1, ), 1, 4, 'discrete'),
        ((2, ), 2, 4, 'discrete'),
        ((1, 1), 1, 4, 'discrete'),
        ((2, 2), 2, 4, 'discrete'),
    ])
    def test_get_actions(self, obs_dim, action_dim, hidden_dim, obs_type):
        if obs_type == 'discrete':
            env = GymEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GymEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=False)
        env.reset()
        obs = env.step(1).observation
        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)
        torch_obs = torch.Tensor(obs)
        actions, _ = policy.get_actions([torch_obs, torch_obs, torch_obs])
        for action in actions:
            assert env.action_space.contains(action)

    # pylint: disable=no-member
    def test_is_pickleable(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      state_include_action=False)

        env.reset()
        obs = env.step(1).observation

        output_action_1, _ = policy.get_action(obs)

        p = cloudpickle.dumps(policy)
        policy_pickled = cloudpickle.loads(p)
        output_action_2, _ = policy_pickled.get_action(obs)

        assert env.action_space.contains(output_action_1)
        assert env.action_space.contains(output_action_2)
        assert output_action_1.shape == output_action_2.shape

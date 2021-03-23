"""Test categoricalCNNPolicy in PyTorch."""
import cloudpickle
import pytest
import torch.nn as nn

from garage.envs import GymEnv
from garage.torch.policies import DiscreteCNNPolicy

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestDiscreteCNNPolicy:

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((1, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (2, ), (0, )),
            ((5, ), (12, ), (1, ), (2, )),
            ((1, 1), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_get_action(self, kernel_sizes, hidden_channels, strides,
                        paddings):
        """Test get_action function."""
        env = GymEnv(DummyDiscretePixelEnv())
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   image_format='NHWC',
                                   hidden_channels=hidden_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   paddings=paddings,
                                   padding_mode='zeros',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)
        env.reset()
        obs = env.step(1).observation

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(int(action[0]))

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((1, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (2, ), (0, )),
            ((5, ), (12, ), (1, ), (2, )),
            ((1, 1), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_get_actions(self, kernel_sizes, hidden_channels, strides,
                         paddings):
        """Test get_actions function."""
        env = GymEnv(DummyDiscretePixelEnv())
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   image_format='NHWC',
                                   hidden_channels=hidden_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   paddings=paddings,
                                   padding_mode='zeros',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        env.reset()
        obs = env.step(1).observation

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(int(action[0]))

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((1, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (2, ), (0, )),
            ((5, ), (12, ), (1, ), (2, )),
            ((1, 1), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_is_pickleable(self, kernel_sizes, hidden_channels, strides,
                           paddings):
        """Test if policy is pickable."""
        env = GymEnv(DummyDiscretePixelEnv())
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   image_format='NHWC',
                                   hidden_channels=hidden_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   paddings=paddings,
                                   padding_mode='zeros',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)
        env.reset()
        obs = env.step(1).observation

        output_action_1, _ = policy.get_action(obs.flatten())

        p = cloudpickle.dumps(policy)
        policy_pickled = cloudpickle.loads(p)
        output_action_2, _ = policy_pickled.get_action(obs)

        assert env.action_space.contains(int(output_action_1[0]))
        assert env.action_space.contains(int(output_action_2[0]))
        assert output_action_1.shape == output_action_2.shape

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((1, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (2, ), (0, )),
            ((5, ), (12, ), (1, ), (2, )),
            ((1, 1), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_obs_unflattened(self, kernel_sizes, hidden_channels, strides,
                             paddings):
        """Test if a flattened image obs is passed to get_action
           then it is unflattened.
        """
        env = GymEnv(DummyDiscretePixelEnv())
        env.reset()
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   image_format='NHWC',
                                   hidden_channels=hidden_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   paddings=paddings,
                                   padding_mode='zeros',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        obs = env.observation_space.sample()
        action, _ = policy.get_action(env.observation_space.flatten(obs))
        env.step(action[0])

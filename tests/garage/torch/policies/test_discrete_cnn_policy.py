"""Test categoricalCNNPolicy in PyTorch."""
import cloudpickle
import pytest
import torch.nn as nn

from garage.envs import GymEnv
from garage.torch import TransposeImage
from garage.torch.policies import DiscreteCNNPolicy

from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestCategoricalCNNPolicy:

    def _initialize_obs_env(self, env):
        """Initialize observation env depends on observation space type.

        If observation space (i.e. akro.Image, gym.spaces.Box) is an image,
        wrap the input of shape (W, H, 3) for PyTorch (N, 3, W, H).

        Return:
            Transformed environment (garage.envs).
        """
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)
        return env

    @pytest.mark.parametrize(
        'action_dim, kernel_sizes, hidden_channels, strides, paddings', [
            (3, (1, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (2, ), (0, )),
            (3, (5, ), (12, ), (1, ), (2, )),
            (3, (1, 1), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_get_action(self, action_dim, kernel_sizes, hidden_channels,
                        strides, paddings):
        """Test get_action function."""
        batch_size = 64
        input_width = 32
        input_height = 32
        in_channel = 3
        input_shape = (batch_size, in_channel, input_height, input_width)
        env = GymEnv(
            DummyDiscreteEnv(obs_dim=input_shape, action_dim=action_dim))

        env = self._initialize_obs_env(env)
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   hidden_channels=hidden_channels,
                                   hidden_sizes=hidden_channels,
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
        assert env.action_space.n == action_dim

    @pytest.mark.parametrize(
        'action_dim, kernel_sizes, hidden_channels, strides, paddings', [
            (3, (1, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (2, ), (0, )),
            (3, (5, ), (12, ), (1, ), (2, )),
            (3, (1, 1), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_get_actions(self, action_dim, kernel_sizes, hidden_channels,
                         strides, paddings):
        """Test get_actions function."""
        batch_size = 64
        input_width = 32
        input_height = 32
        in_channel = 3
        input_shape = (batch_size, in_channel, input_height, input_width)
        env = GymEnv(
            DummyDiscreteEnv(obs_dim=input_shape, action_dim=action_dim))

        env = self._initialize_obs_env(env)
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   hidden_channels=hidden_channels,
                                   hidden_sizes=hidden_channels,
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
            assert env.action_space.n == action_dim

    @pytest.mark.parametrize(
        'action_dim, kernel_sizes, hidden_channels, strides, paddings', [
            (3, (1, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (2, ), (0, )),
            (3, (5, ), (12, ), (1, ), (2, )),
            (3, (1, 1), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_is_pickleable(self, action_dim, kernel_sizes, hidden_channels,
                           strides, paddings):
        """Test if policy is pickable."""
        batch_size = 64
        input_width = 32
        input_height = 32
        in_channel = 3
        input_shape = (batch_size, in_channel, input_height, input_width)
        env = GymEnv(
            DummyDiscreteEnv(obs_dim=input_shape, action_dim=action_dim))

        env = self._initialize_obs_env(env)
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   hidden_channels=hidden_channels,
                                   hidden_sizes=hidden_channels,
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
        'action_dim, kernel_sizes, hidden_channels, strides, paddings', [
            (3, (1, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (1, ), (0, )),
            (3, (3, ), (32, ), (2, ), (0, )),
            (3, (5, ), (12, ), (1, ), (2, )),
            (3, (1, 1), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (1, 1), (0, 0)),
            (3, (3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_obs_unflattened(self, action_dim, kernel_sizes, hidden_channels,
                             strides, paddings):
        """Test if a flattened image obs is passed to get_action
           then it is unflattened.
        """
        batch_size = 64
        input_width = 32
        input_height = 32
        in_channel = 3
        input_shape = (batch_size, in_channel, input_height, input_width)
        env = GymEnv(
            DummyDiscreteEnv(obs_dim=input_shape, action_dim=action_dim))
        env = self._initialize_obs_env(env)

        env.reset()
        policy = DiscreteCNNPolicy(env_spec=env.spec,
                                   hidden_channels=hidden_channels,
                                   hidden_sizes=hidden_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   paddings=paddings,
                                   padding_mode='zeros',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        obs = env.observation_space.sample()
        action, _ = policy.get_action(env.observation_space.flatten(obs))
        env.step(action)

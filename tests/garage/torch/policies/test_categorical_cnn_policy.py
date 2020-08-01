"""Test categoricalCNNPolicy in PyTorch."""
import cloudpickle
import pytest

from garage.envs import GarageEnv
from garage.torch import TransposeImage
from garage.torch.policies import CategoricalCNNPolicy

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


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
        'hidden_channels, kernel_sizes, strides, hidden_sizes', [
            ((3, ), (3, ), (1, ), (4, )),
            ((3, 3), (3, 3), (1, 1), (4, 4)),
            ((3, 3), (3, 3), (2, 2), (4, 4)),
        ])
    def test_get_action(self, hidden_channels, kernel_sizes, strides,
                        hidden_sizes):
        """Test get_action function."""
        env = GarageEnv(DummyDiscretePixelEnv())
        env = self._initialize_obs_env(env)
        policy = CategoricalCNNPolicy(env=env,
                                      kernel_sizes=kernel_sizes,
                                      hidden_channels=hidden_channels,
                                      strides=strides,
                                      hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

    @pytest.mark.parametrize(
        'hidden_channels, kernel_sizes, strides, hidden_sizes', [
            ((3, ), (3, ), (1, ), (4, )),
            ((3, 3), (3, 3), (1, 1), (4, 4)),
            ((3, 3), (3, 3), (2, 2), (4, 4)),
        ])
    def test_get_action_img_obs(self, hidden_channels, kernel_sizes, strides,
                                hidden_sizes):
        """Test get_action function with akro.Image observation space."""
        env = GarageEnv(DummyDiscretePixelEnv(), is_image=True)
        env = self._initialize_obs_env(env)
        policy = CategoricalCNNPolicy(env=env,
                                      kernel_sizes=kernel_sizes,
                                      hidden_channels=hidden_channels,
                                      strides=strides,
                                      hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

    @pytest.mark.parametrize(
        'hidden_channels, kernel_sizes, strides, hidden_sizes', [
            ((3, ), (3, ), (1, ), (4, )),
            ((3, 3), (3, 3), (1, 1), (4, 4)),
            ((3, 3), (3, 3), (2, 2), (4, 4)),
        ])
    def test_get_actions(self, hidden_channels, kernel_sizes, strides,
                         hidden_sizes):
        """Test get_actions function with akro.Image observation space."""
        env = GarageEnv(DummyDiscretePixelEnv(), is_image=True)
        env = self._initialize_obs_env(env)
        policy = CategoricalCNNPolicy(env=env,
                                      kernel_sizes=kernel_sizes,
                                      hidden_channels=hidden_channels,
                                      strides=strides,
                                      hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize(
        'hidden_channels, kernel_sizes, strides, hidden_sizes', [
            ((3, ), (3, ), (1, ), (4, )),
            ((3, 3), (3, 3), (1, 1), (4, 4)),
            ((3, 3), (3, 3), (2, 2), (4, 4)),
        ])
    def test_is_pickleable(self, hidden_channels, kernel_sizes, strides,
                           hidden_sizes):
        """Test if policy is pickable."""
        env = GarageEnv(DummyDiscretePixelEnv(), is_image=True)
        env = self._initialize_obs_env(env)
        policy = CategoricalCNNPolicy(env=env,
                                      kernel_sizes=kernel_sizes,
                                      hidden_channels=hidden_channels,
                                      strides=strides,
                                      hidden_sizes=hidden_sizes)
        env.reset()
        obs, _, _, _ = env.step(1)

        output_action_1, _ = policy.get_action(obs)

        p = cloudpickle.dumps(policy)
        policy_pickled = cloudpickle.loads(p)
        output_action_2, _ = policy_pickled.get_action(obs)

        assert env.action_space.contains(output_action_1)
        assert env.action_space.contains(output_action_2)
        assert output_action_1.shape == output_action_2.shape

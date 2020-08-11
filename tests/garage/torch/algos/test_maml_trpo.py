"""This script is a test that fails when MAML-TRPO performance is too low."""
import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config
from tests.fixtures.envs.dummy import DummyMultiTaskBoxEnv

try:
    # pylint: disable=unused-import
    import mujoco_py  # noqa: F401
except ImportError:
    pytest.skip('To use mujoco-based features, please install garage[mujoco].',
                allow_module_level=True)
except Exception:  # pylint: disable=broad-except
    pytest.skip(
        'Skipping tests, failed to import mujoco. Do you have a '
        'valid mujoco key installed?',
        allow_module_level=True)

from garage.envs.mujoco import HalfCheetahDirEnv  # isort:skip # pylint: disable=ungrouped-imports # noqa: E501


@pytest.mark.mujoco
def test_maml_trpo_pendulum():
    """Test PPO with Pendulum environment."""
    env = normalize(GymEnv(HalfCheetahDirEnv()), expected_action_scale=10.)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32))

    episodes_per_task = 5
    max_episode_length = 100

    runner = LocalRunner(snapshot_config)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=value_function,
                    max_episode_length=max_episode_length,
                    meta_batch_size=5,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1)

    runner.setup(algo, env, sampler_cls=LocalSampler)
    last_avg_ret = runner.train(n_epochs=5,
                                batch_size=episodes_per_task *
                                max_episode_length)

    assert last_avg_ret > -5

    env.close()


def test_maml_trpo_dummy_named_env():
    """Test with dummy environment that has env_name."""
    env = normalize(GymEnv(DummyMultiTaskBoxEnv()), expected_action_scale=10.)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32))

    episodes_per_task = 2
    max_episode_length = 100

    runner = LocalRunner(snapshot_config)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=value_function,
                    max_episode_length=max_episode_length,
                    meta_batch_size=5,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1)

    runner.setup(algo, env, sampler_cls=LocalSampler)
    runner.train(n_epochs=2, batch_size=episodes_per_task * max_episode_length)

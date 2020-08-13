"""This script is a test that fails when MAML-TRPO performance is too low."""
import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config

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

from garage.envs.mujoco import HalfCheetahDirEnv  # isort:skip


@pytest.mark.mujoco
class TestMAMLPPO:
    """Test class for MAML-PPO."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = normalize(GymEnv(HalfCheetahDirEnv()),
                             expected_action_scale=10.)
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.value_function = GaussianMLPValueFunction(env_spec=self.env.spec,
                                                       hidden_sizes=(32, 32))

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    def test_ppo_pendulum(self):
        """Test PPO with Pendulum environment."""
        deterministic.set_seed(0)

        episodes_per_task = 5
        max_episode_length = 100

        runner = LocalRunner(snapshot_config)
        algo = MAMLPPO(env=self.env,
                       policy=self.policy,
                       value_function=self.value_function,
                       max_episode_length=max_episode_length,
                       meta_batch_size=5,
                       discount=0.99,
                       gae_lambda=1.,
                       inner_lr=0.1,
                       num_grad_updates=1)

        runner.setup(algo, self.env, sampler_cls=LocalSampler)
        last_avg_ret = runner.train(n_epochs=10,
                                    batch_size=episodes_per_task *
                                    max_episode_length)

        assert last_avg_ret > -5

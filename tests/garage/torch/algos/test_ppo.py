"""This script creates a test that fails when PPO performance is too low."""
import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config


class TestPPO:
    """Test class for PPO."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = normalize(GymEnv('InvertedDoublePendulum-v2'))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.value_function = GaussianMLPValueFunction(env_spec=self.env.spec)

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    @pytest.mark.mujoco
    def test_ppo_pendulum(self):
        """Test PPO with Pendulum environment."""
        deterministic.set_seed(0)

        runner = LocalRunner(snapshot_config)
        algo = PPO(env_spec=self.env.spec,
                   policy=self.policy,
                   value_function=self.value_function,
                   max_episode_length=100,
                   discount=0.99,
                   gae_lambda=0.97,
                   lr_clip_range=2e-1)

        runner.setup(algo, self.env, sampler_cls=LocalSampler)
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

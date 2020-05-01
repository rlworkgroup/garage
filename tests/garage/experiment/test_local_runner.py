import gym
import pytest
import torch

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.plotter import Plotter
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from tests.fixtures import snapshot_config


class TestLocalRunner:
    """Test class for LocalRunner."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
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
    def test_set_plot(self):
        deterministic.set_seed(0)

        runner = LocalRunner(snapshot_config)
        algo = PPO(env_spec=self.env.spec,
                   policy=self.policy,
                   value_function=self.value_function,
                   max_path_length=100,
                   discount=0.99,
                   gae_lambda=0.97,
                   lr_clip_range=2e-1)

        runner.setup(algo, self.env)
        runner.train(n_epochs=1, batch_size=100, plot=True)

        assert isinstance(
            runner._plotter,
            Plotter), ('self.plotter in LocalRunner should be set to Plotter.')

import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.plotter import Plotter
from garage.sampler import LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


class TestTrainer:
    """Test class for Trainer."""

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
        deterministic.set_seed(0)
        self.sampler = LocalSampler(
            agents=self.policy,
            envs=self.env,
            max_episode_length=self.env.spec.max_episode_length,
            is_tf_worker=True)

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    @pytest.mark.mujoco
    def test_set_plot(self):
        trainer = Trainer(snapshot_config)
        algo = PPO(env_spec=self.env.spec,
                   policy=self.policy,
                   value_function=self.value_function,
                   sampler=self.sampler,
                   discount=0.99,
                   gae_lambda=0.97,
                   lr_clip_range=2e-1)

        trainer.setup(algo, self.env)
        trainer.train(n_epochs=1, batch_size=100, plot=True)

        assert isinstance(
            trainer._plotter,
            Plotter), ('self.plotter in Trainer should be set to Plotter.')


def test_setup_no_sampler():
    trainer = Trainer(snapshot_config)

    class SupervisedAlgo:

        def train(self, trainer):
            # pylint: disable=undefined-loop-variable
            for epoch in trainer.step_epochs():
                pass
            assert epoch == 4

    trainer.setup(SupervisedAlgo(), None)
    trainer.train(n_epochs=5)

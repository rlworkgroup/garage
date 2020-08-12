import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.plotter import Plotter
from garage.sampler import LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config


class TestLocalRunner:
    """Test class for LocalRunner."""

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
    def test_set_plot(self):
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
        runner.train(n_epochs=1, batch_size=100, plot=True)

        assert isinstance(
            runner._plotter,
            Plotter), ('self.plotter in LocalRunner should be set to Plotter.')


def test_setup_no_sampler():
    runner = LocalRunner(snapshot_config)

    class SupervisedAlgo:

        def train(self, runner):
            # pylint: disable=undefined-loop-variable
            for epoch in runner.step_epochs():
                pass
            assert epoch == 4

    runner.setup(SupervisedAlgo(), None)
    runner.train(n_epochs=5)


class CrashingAlgo:

    def train(self, runner):
        # pylint: disable=undefined-loop-variable
        for epoch in runner.step_epochs():
            runner.obtain_samples(epoch)


def test_setup_no_sampler_cls():
    runner = LocalRunner(snapshot_config)
    algo = CrashingAlgo()
    algo.max_episode_length = 100
    runner.setup(algo, None)
    with pytest.raises(ValueError, match='sampler_cls'):
        runner.train(n_epochs=5)


def test_setup_no_policy():
    runner = LocalRunner(snapshot_config)
    with pytest.raises(ValueError, match='policy'):
        runner.setup(CrashingAlgo(), None, sampler_cls=LocalSampler)


def test_setup_no_max_episode_length():
    runner = LocalRunner(snapshot_config)
    algo = CrashingAlgo()
    algo.policy = ()
    with pytest.raises(ValueError, match='max_episode_length'):
        runner.setup(algo, None, sampler_cls=LocalSampler)

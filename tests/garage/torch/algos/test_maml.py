"""Unit tests of MAML."""
from functools import partial

import pytest
import torch

from garage.envs import HalfCheetahDirEnv, normalize
from garage.envs.base import GarageEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy


@pytest.fixture
def set_params():  # noqa: D202
    """Set parameters of a module."""

    def _set_params(v, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.fill_(v)
            m.bias.data.fill_(v)

    return _set_params


@pytest.fixture
def test_params():  # noqa: D202
    """Test parameters equal to a value."""

    def _test_params(v, m):
        if isinstance(m, torch.nn.Linear):
            assert torch.all(torch.eq(m.weight.data, v))
            assert torch.all(torch.eq(m.bias.data, v))

    return _test_params


class TestMAML:
    """Test class for MAML."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = GarageEnv(
            normalize(HalfCheetahDirEnv(), expected_action_scale=10.))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.baseline = LinearFeatureBaseline(env_spec=self.env.spec)
        self.algo = MAMLPPO(env=self.env,
                            policy=self.policy,
                            baseline=self.baseline,
                            max_path_length=100,
                            meta_batch_size=5,
                            discount=0.99,
                            gae_lambda=1.,
                            inner_lr=0.1,
                            num_grad_updates=1)

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    def test_get_exploration_policy(self, set_params, test_params):
        """Test if an independent copy of policy is returned."""
        self.policy.apply(partial(set_params, 0.1))
        adapt_policy = self.algo.get_exploration_policy()
        adapt_policy.apply(partial(set_params, 0.2))

        # Old policy should remain untouched
        self.policy.apply(partial(test_params, 0.1))
        adapt_policy.apply(partial(test_params, 0.2))

    def test_adapt_policy(self, set_params, test_params):
        """Test if policy can adapt to samples."""
        worker = WorkerFactory(seed=100, max_path_length=100)
        sampler = LocalSampler.from_worker_factory(worker, self.policy,
                                                   self.env)

        self.policy.apply(partial(set_params, 0.1))
        adapt_policy = self.algo.get_exploration_policy()
        trajs = sampler.obtain_samples(0, 100, adapt_policy)
        self.algo.adapt_policy(adapt_policy, trajs)

        # Old policy should remain untouched
        self.policy.apply(partial(test_params, 0.1))

        # Adapted policy should not be identical to old policy
        for v1, v2 in zip(adapt_policy.parameters(), self.policy.parameters()):
            if v1.data.ne(v2.data).sum() > 0:
                break
        else:
            pytest.fail('Parameters of adapted policy should not be '
                        'identical to the old policy.')

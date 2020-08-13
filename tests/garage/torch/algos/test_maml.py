"""Unit tests of MAML."""
from functools import partial

import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

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
class TestMAML:
    """Test class for MAML."""

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
        self.algo = MAMLPPO(env=self.env,
                            policy=self.policy,
                            value_function=self.value_function,
                            max_episode_length=100,
                            meta_batch_size=5,
                            discount=0.99,
                            gae_lambda=1.,
                            inner_lr=0.1,
                            num_grad_updates=1)

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    @staticmethod
    def _set_params(v, m):
        """Set the parameters of a module to a value."""
        if isinstance(m, torch.nn.Linear):
            m.weight.data.fill_(v)
            m.bias.data.fill_(v)

    @staticmethod
    def _test_params(v, m):
        """Test if all parameters of a module equal to a value."""
        if isinstance(m, torch.nn.Linear):
            assert torch.all(torch.eq(m.weight.data, v))
            assert torch.all(torch.eq(m.bias.data, v))

    def test_get_exploration_policy(self):
        """Test if an independent copy of policy is returned."""
        self.policy.apply(partial(self._set_params, 0.1))
        adapt_policy = self.algo.get_exploration_policy()
        adapt_policy.apply(partial(self._set_params, 0.2))

        # Old policy should remain untouched
        self.policy.apply(partial(self._test_params, 0.1))
        adapt_policy.apply(partial(self._test_params, 0.2))

    def test_adapt_policy(self):
        """Test if policy can adapt to samples."""
        worker = WorkerFactory(seed=100, max_episode_length=100)
        sampler = LocalSampler.from_worker_factory(worker, self.policy,
                                                   self.env)

        self.policy.apply(partial(self._set_params, 0.1))
        adapt_policy = self.algo.get_exploration_policy()
        eps = sampler.obtain_samples(0, 100, adapt_policy)
        self.algo.adapt_policy(adapt_policy, eps)

        # Old policy should remain untouched
        self.policy.apply(partial(self._test_params, 0.1))

        # Adapted policy should not be identical to old policy
        for v1, v2 in zip(adapt_policy.parameters(), self.policy.parameters()):
            if v1.data.ne(v2.data).sum() > 0:
                break
        else:
            pytest.fail('Parameters of adapted policy should not be '
                        'identical to the old policy.')

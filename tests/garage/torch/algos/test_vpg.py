"""This script creates a test that fails when VPG performance is too low."""
import pytest
import torch

from garage.envs import GymEnv
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config

# yapf: disable
INVALID_ENTROPY_CONFIG = [
    ({'entropy_method': 'INVALID_ENTROPY_METHOD'},
        ValueError, 'entropy_method'),
    ({'entropy_method': 'max',
      'center_adv': True},
        ValueError, 'center_adv'),
    ({'entropy_method': 'max',
      'center_adv': False,
      'stop_entropy_gradient': False},
        ValueError, 'entropy_method'),
    ({'entropy_method': 'no_entropy',
      'policy_ent_coeff': 1.0},
        ValueError, 'policy_ent_coeff')
]
# yapf: enable


class TestVPG:
    """Test class for VPG."""

    @classmethod
    def setup_class(cls):
        """Setup method which is called once before all tests in this class."""
        deterministic.set_seed(0)

    def setup_method(self):
        """Setup method which is called before every test."""
        self._env = GymEnv('InvertedDoublePendulum-v2')
        self._runner = LocalRunner(snapshot_config)

        self._policy = GaussianMLPPolicy(env_spec=self._env.spec,
                                         hidden_sizes=[64, 64],
                                         hidden_nonlinearity=torch.tanh,
                                         output_nonlinearity=None)
        self._params = {
            'env_spec': self._env.spec,
            'policy': self._policy,
            'value_function':
            GaussianMLPValueFunction(env_spec=self._env.spec),
            'max_episode_length': 100,
            'discount': 0.99,
        }

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self._env.close()

    @pytest.mark.mujoco
    def test_vpg_no_entropy(self):
        """Test VPG with no_entropy."""
        self._params['positive_adv'] = True
        self._params['use_softplus_entropy'] = True

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env, sampler_cls=LocalSampler)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

    @pytest.mark.mujoco
    def test_vpg_max(self):
        """Test VPG with maximum entropy."""
        self._params['center_adv'] = False
        self._params['stop_entropy_gradient'] = True
        self._params['entropy_method'] = 'max'

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env, sampler_cls=LocalSampler)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

    @pytest.mark.mujoco
    def test_vpg_regularized(self):
        """Test VPG with entropy_regularized."""
        self._params['entropy_method'] = 'regularized'

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env, sampler_cls=LocalSampler)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

    @pytest.mark.mujoco
    @pytest.mark.parametrize('algo_param, error, msg', INVALID_ENTROPY_CONFIG)
    def test_invalid_entropy_config(self, algo_param, error, msg):
        """Test VPG with invalid entropy config."""
        self._params.update(algo_param)
        with pytest.raises(error, match=msg):
            VPG(**self._params)

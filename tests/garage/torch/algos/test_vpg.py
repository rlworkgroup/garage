import gym
import pytest
import torch

from garage.envs.base import GarageEnv
from garage.experiment import deterministic
from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy
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

    @classmethod
    def setup_class(cls):
        deterministic.set_seed(0)

    def setup_method(self):
        self._env = GarageEnv(gym.make('InvertedDoublePendulum-v2'))
        self._runner = LocalRunner(snapshot_config)

        policy = GaussianMLPPolicy(env_spec=self._env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)
        self._params = {
            'env_spec': self._env.spec,
            'policy': policy,
            'optimizer': torch.optim.Adam,
            'baseline': LinearFeatureBaseline(env_spec=self._env.spec),
            'max_path_length': 100,
            'discount': 0.99,
            'policy_lr': 1e-2
        }

    def teardown_method(self):
        self._env.close()

    def test_vpg_no_entropy(self):
        """Test VPG with no_entropy."""
        self._params['positive_adv'] = True
        self._params['use_softplus_entropy'] = True

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

    def test_vpg_max(self):
        """Test VPG with maximum entropy."""
        self._params['center_adv'] = False
        self._params['stop_entropy_gradient'] = True
        self._params['entropy_method'] = 'max'

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

    def test_vpg_regularized(self):
        """Test VPG with entropy_regularized."""
        self._params['entropy_method'] = 'regularized'

        algo = VPG(**self._params)
        self._runner.setup(algo, self._env)
        last_avg_ret = self._runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 30

    @pytest.mark.parametrize('algo_param, error, msg', INVALID_ENTROPY_CONFIG)
    def test_invalid_entropy_config(self, algo_param, error, msg):
        self._params.update(algo_param)
        with pytest.raises(error, match=msg):
            VPG(**self._params)

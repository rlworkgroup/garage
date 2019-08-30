"""
This script creates a test that fails when garage.tf.algos.TRPO performance is
too low.
"""
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTRPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        self.baseline = GaussianMLPBaseline(
            env_spec=self.env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )

    @pytest.mark.large
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_path_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 50

    def test_trpo_unknown_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with pytest.raises(ValueError, match='Invalid kl_constraint'):
            TRPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0,
                kl_constraint='random kl_constraint',
            )

    @pytest.mark.large
    def test_trpo_soft_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_path_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0,
                        kl_constraint='soft')
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 45

    def teardown_method(self):
        self.env.close()
        super().teardown_method()

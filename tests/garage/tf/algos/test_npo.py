"""
This script creates a test that fails when garage.tf.algos.NPO performance is
too low.
"""
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.tf.algos import NPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestNPO(TfGraphTestCase):

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
    def test_npo_pendulum(self):
        """Test NPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = NPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       gae_lambda=0.98,
                       policy_ent_coeff=0.0)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 20

    def test_npo_with_unknown_pg_loss(self):
        """Test NPO with unkown pg loss."""
        with pytest.raises(ValueError, match='Invalid pg_loss'):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                pg_loss='random pg_loss',
            )

    def test_npo_with_invalid_entropy_method(self):
        """Test NPO with invalid entropy method."""
        with pytest.raises(ValueError, match='Invalid entropy_method'):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                entropy_method=None,
            )

    def test_npo_with_max_entropy_and_center_adv(self):
        """Test NPO with max entropy and center_adv."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                entropy_method='max',
                center_adv=True,
            )

    def test_npo_with_max_entropy_and_no_stop_entropy_gradient(self):
        """Test NPO with max entropy and false stop_entropy_gradient."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                entropy_method='max',
                stop_entropy_gradient=False,
            )

    def test_npo_with_invalid_no_entropy_configuration(self):
        """Test NPO with invalid no entropy configuration."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                entropy_method='no_entropy',
                policy_ent_coeff=0.02,
            )

    def teardown_method(self):
        self.env.close()
        super().teardown_method()

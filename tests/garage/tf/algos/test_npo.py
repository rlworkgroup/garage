"""
This script creates a test that fails when garage.tf.algos.NPO performance is
too low.
"""
import pytest
import tensorflow as tf

from garage.envs import GymEnv, normalize
from garage.sampler import LocalSampler
from garage.tf.algos import NPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestNPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = normalize(
            GymEnv('InvertedDoublePendulum-v2', max_episode_length=100))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        self.baseline = GaussianMLPBaseline(
            env_spec=self.env.spec,
            hidden_sizes=(32, 32),
        )
        self.sampler = LocalSampler(
            agents=self.policy,
            envs=self.env,
            max_episode_length=self.env.spec.max_episode_length,
            is_tf_worker=True)

    @pytest.mark.flaky
    @pytest.mark.mujoco
    def test_npo_pendulum(self):
        """Test NPO with Pendulum environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            algo = NPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       sampler=self.sampler,
                       discount=0.99,
                       gae_lambda=0.98,
                       policy_ent_coeff=0.0)
            trainer.setup(algo, self.env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 20

    @pytest.mark.mujoco
    def test_npo_with_unknown_pg_loss(self):
        """Test NPO with unkown pg loss."""
        with pytest.raises(ValueError, match='Invalid pg_loss'):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                pg_loss='random pg_loss',
            )

    @pytest.mark.mujoco
    def test_npo_with_invalid_entropy_method(self):
        """Test NPO with invalid entropy method."""
        with pytest.raises(ValueError, match='Invalid entropy_method'):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                entropy_method=None,
            )

    @pytest.mark.mujoco
    def test_npo_with_max_entropy_and_center_adv(self):
        """Test NPO with max entropy and center_adv."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                entropy_method='max',
                center_adv=True,
            )

    @pytest.mark.mujoco
    def test_npo_with_max_entropy_and_no_stop_entropy_gradient(self):
        """Test NPO with max entropy and false stop_entropy_gradient."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                entropy_method='max',
                stop_entropy_gradient=False,
            )

    @pytest.mark.mujoco
    def test_npo_with_invalid_no_entropy_configuration(self):
        """Test NPO with invalid no entropy configuration."""
        with pytest.raises(ValueError):
            NPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                entropy_method='no_entropy',
                policy_ent_coeff=0.02,
            )

    def teardown_method(self):
        self.env.close()
        super().teardown_method()

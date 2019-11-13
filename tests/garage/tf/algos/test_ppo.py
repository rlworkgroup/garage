"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.baselines import ContinuousMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase
from tests.fixtures.envs.wrappers import ReshapeObservation


class TestPPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        self.lstm_policy = GaussianLSTMPolicy(env_spec=self.env.spec)
        self.gru_policy = GaussianGRUPolicy(env_spec=self.env.spec)
        self.baseline = GaussianMLPBaseline(
            env_spec=self.env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )

    # large marker removed to balance test jobs
    # @pytest.mark.large
    def test_ppo_pendulum(self):
        """Test PPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = PPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       lr_clip_range=0.01,
                       optimizer_args=dict(batch_size=32, max_epochs=10))
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 35

    # large marker removed to balance test jobs
    # @pytest.mark.large
    def test_ppo_with_maximum_entropy(self):
        """Test PPO with maxium entropy method."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = PPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       lr_clip_range=0.01,
                       optimizer_args=dict(batch_size=32, max_epochs=10),
                       stop_entropy_gradient=True,
                       entropy_method='max',
                       policy_ent_coeff=0.02,
                       center_adv=False)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 35

    # large marker removed to balance test jobs
    # @pytest.mark.large
    def test_ppo_with_neg_log_likeli_entropy_estimation_and_max(self):
        """
        Test PPO with negative log likelihood entropy estimation and max
        entropy method.
        """
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = PPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       lr_clip_range=0.01,
                       optimizer_args=dict(batch_size=32, max_epochs=10),
                       stop_entropy_gradient=True,
                       use_neg_logli_entropy=True,
                       entropy_method='max',
                       policy_ent_coeff=0.02,
                       center_adv=False)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 35

    # large marker removed to balance test jobs
    # @pytest.mark.large
    def test_ppo_with_neg_log_likeli_entropy_estimation_and_regularized(self):
        """
        Test PPO with negative log likelihood entropy estimation and
        regularized entropy method.
        """
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = PPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       lr_clip_range=0.01,
                       optimizer_args=dict(batch_size=32, max_epochs=10),
                       stop_entropy_gradient=True,
                       use_neg_logli_entropy=True,
                       entropy_method='regularized',
                       policy_ent_coeff=0.0,
                       center_adv=True)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 35

    # large marker removed to balance test jobs
    # @pytest.mark.large
    def test_ppo_with_regularized_entropy(self):
        """Test PPO with regularized entropy method."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = PPO(env_spec=self.env.spec,
                       policy=self.policy,
                       baseline=self.baseline,
                       max_path_length=100,
                       discount=0.99,
                       lr_clip_range=0.01,
                       optimizer_args=dict(batch_size=32, max_epochs=10),
                       stop_entropy_gradient=False,
                       entropy_method='regularized',
                       policy_ent_coeff=0.02,
                       center_adv=True,
                       flatten_input=False)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 35

    def test_ppo_pendulum_flatten_input(self):
        """Test PPO with CartPole to test observation flattening."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(
                normalize(ReshapeObservation(gym.make('CartPole-v1'), (2, 2))))
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_nonlinearity=tf.nn.tanh,
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = PPO(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=100,
                       discount=0.99,
                       gae_lambda=0.95,
                       lr_clip_range=0.2,
                       policy_ent_coeff=0.0,
                       optimizer_args=dict(
                           batch_size=32,
                           max_epochs=10,
                           tf_optimizer_args=dict(learning_rate=1e-3),
                       ))
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80

    def teardown_method(self):
        self.env.close()
        super().teardown_method()


class TestPPOContinuousBaseline(TfGraphTestCase):

    @pytest.mark.huge
    def test_ppo_pendulum_continuous_baseline(self):
        """Test PPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(64, 64),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )
            baseline = ContinuousMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 100

            env.close()

    @pytest.mark.large
    def test_ppo_pendulum_recurrent_continuous_baseline(self):
        """Test PPO with Pendulum environment and recurrent policy."""
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            policy = GaussianLSTMPolicy(env_spec=env.spec, )
            baseline = ContinuousMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 100

            env.close()


class TestPPOPendulumLSTM(TfGraphTestCase):

    @pytest.mark.large
    def test_ppo_pendulum_lstm(self):
        """Test PPO with Pendulum environment and recurrent policy."""
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            lstm_policy = GaussianLSTMPolicy(env_spec=env.spec)
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=lstm_policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80


class TestPPOPendulumGRU(TfGraphTestCase):

    @pytest.mark.large
    def test_ppo_pendulum_gru(self):
        """Test PPO with Pendulum environment and recurrent policy."""
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            gru_policy = GaussianGRUPolicy(env_spec=env.spec)
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=gru_policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80

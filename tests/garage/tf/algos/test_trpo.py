"""
This script creates a test that fails when garage.tf.algos.TRPO performance is
too low.
"""
# yapf: disable
import pytest
import tensorflow as tf

# yapf: disable
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, snapshotter
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianCNNBaseline, GaussianMLPBaseline
from garage.tf.optimizers import FiniteDifferenceHVP
from garage.tf.policies import (CategoricalCNNPolicy, CategoricalGRUPolicy,
                                CategoricalLSTMPolicy, GaussianMLPPolicy)
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable


class TestTRPO(TfGraphTestCase):

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

    @pytest.mark.mujoco_long
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        sampler=self.sampler,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0)
            trainer.setup(algo, self.env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

    @pytest.mark.mujoco
    def test_trpo_unknown_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with pytest.raises(ValueError, match='Invalid kl_constraint'):
            TRPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0,
                kl_constraint='random kl_constraint',
            )

    @pytest.mark.mujoco_long
    def test_trpo_soft_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        sampler=self.sampler,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0,
                        kl_constraint='soft')
            trainer.setup(algo, self.env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 45

    @pytest.mark.mujoco_long
    def test_trpo_lstm_cartpole(self):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(GymEnv('CartPole-v1', max_episode_length=100))

            policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                            base_eps=1e-5)))

            snapshotter.snapshot_dir = './'
            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 60

            env.close()

    @pytest.mark.mujoco_long
    def test_trpo_gru_cartpole(self):
        deterministic.set_seed(2)
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(GymEnv('CartPole-v1', max_episode_length=100))

            policy = CategoricalGRUPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                            base_eps=1e-5)))

            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

            env.close()

    def teardown_method(self):
        self.env.close()
        super().teardown_method()


class TestTRPOCNNCubeCrash(TfGraphTestCase):

    @pytest.mark.large
    def test_trpo_cnn_cubecrash(self):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(GymEnv('CubeCrash-v0', max_episode_length=100))

            policy = CategoricalCNNPolicy(env_spec=env.spec,
                                          filters=((32, (8, 8)), (64, (4, 4))),
                                          strides=(4, 2),
                                          padding='VALID',
                                          hidden_sizes=(32, 32))

            baseline = GaussianCNNBaseline(env_spec=env.spec,
                                           filters=((32, (8, 8)), (64, (4,
                                                                        4))),
                                           strides=(4, 2),
                                           padding='VALID',
                                           hidden_sizes=(32, 32),
                                           use_trust_region=True)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=0.99,
                        gae_lambda=0.98,
                        max_kl_step=0.01,
                        policy_ent_coeff=0.0)

            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > -1.5

            env.close()

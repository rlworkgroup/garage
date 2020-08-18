"""
This script creates a test that fails when garage.tf.algos.TRPO performance is
too low.
"""
# yapf: disable
import pytest
import tensorflow as tf

# yapf: disable
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalTFRunner, snapshotter
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianCNNBaseline, GaussianMLPBaseline
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import (CategoricalCNNPolicy,
                                CategoricalGRUPolicy,
                                CategoricalLSTMPolicy,
                                GaussianMLPPolicy)

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable


class TestTRPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = normalize(GymEnv('InvertedDoublePendulum-v2'))
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

    @pytest.mark.mujoco_long
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_episode_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0)
            runner.setup(algo, self.env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

    @pytest.mark.mujoco
    def test_trpo_unknown_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with pytest.raises(ValueError, match='Invalid kl_constraint'):
            TRPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                max_episode_length=100,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0,
                kl_constraint='random kl_constraint',
            )

    @pytest.mark.mujoco_long
    def test_trpo_soft_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_episode_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0,
                        kl_constraint='soft')
            runner.setup(algo, self.env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 45

    @pytest.mark.mujoco_long
    def test_trpo_lstm_cartpole(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = normalize(GymEnv('CartPole-v1'))

            policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_episode_length=100,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                            base_eps=1e-5)))

            snapshotter.snapshot_dir = './'
            runner.setup(algo, env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 60

            env.close()

    @pytest.mark.mujoco_long
    def test_trpo_gru_cartpole(self):
        deterministic.set_seed(2)
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = normalize(GymEnv('CartPole-v1'))

            policy = CategoricalGRUPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_episode_length=100,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                            base_eps=1e-5)))

            runner.setup(algo, env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

            env.close()

    def teardown_method(self):
        self.env.close()
        super().teardown_method()


class TestTRPOCNNCubeCrash(TfGraphTestCase):

    @pytest.mark.large
    def test_trpo_cnn_cubecrash(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = normalize(GymEnv('CubeCrash-v0'))

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

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_episode_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        max_kl_step=0.01,
                        policy_ent_coeff=0.0)

            runner.setup(algo, env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > -1.5

            env.close()

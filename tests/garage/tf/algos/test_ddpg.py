"""This script creates a test that fails when garage.tf.algos.DDPG performance
is too low.
"""
import pytest
import tensorflow as tf

from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import get_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestDDPG(TfGraphTestCase):
    """Tests for DDPG algorithm."""

    @pytest.mark.mujoco_long
    def test_ddpg_double_pendulum(self):
        """Test DDPG with Pendulum environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = GymEnv('InvertedDoublePendulum-v2')
            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[64, 64],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)
            exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                           policy,
                                                           sigma=0.2)
            qf = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=tf.nn.relu)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e5))
            sampler = LocalSampler(
                agents=exploration_policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True,
                worker_class=FragmentWorker)
            algo = DDPG(
                env_spec=env.spec,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                replay_buffer=replay_buffer,
                sampler=sampler,
                steps_per_epoch=20,
                target_update_tau=1e-2,
                n_train_steps=50,
                discount=0.9,
                min_buffer_size=int(5e3),
                exploration_policy=exploration_policy,
            )
            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
            assert last_avg_ret > 60

            env.close()

    @pytest.mark.mujoco_long
    def test_ddpg_pendulum(self):
        """Test DDPG with Pendulum environment.

        This environment has a [-3, 3] action_space bound.
        """
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(
                GymEnv('InvertedPendulum-v2', max_episode_length=100))
            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[64, 64],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)
            exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                           policy,
                                                           sigma=0.2)
            qf = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=tf.nn.relu)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
            sampler = LocalSampler(
                agents=exploration_policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True,
                worker_class=FragmentWorker)
            algo = DDPG(
                env_spec=env.spec,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                replay_buffer=replay_buffer,
                sampler=sampler,
                steps_per_epoch=20,
                target_update_tau=1e-2,
                n_train_steps=50,
                discount=0.9,
                min_buffer_size=int(5e3),
                exploration_policy=exploration_policy,
            )
            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
            assert last_avg_ret > 10

            env.close()

    @pytest.mark.mujoco_long
    def test_ddpg_pendulum_with_decayed_weights(self):
        """Test DDPG with Pendulum environment and decayed weights.

        This environment has a [-3, 3] action_space bound.
        """
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(
                GymEnv('InvertedPendulum-v2', max_episode_length=100))
            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[64, 64],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)
            exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                           policy,
                                                           sigma=0.2)
            qf = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=tf.nn.relu)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
            sampler = LocalSampler(
                agents=exploration_policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True,
                worker_class=FragmentWorker)
            algo = DDPG(
                env_spec=env.spec,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                replay_buffer=replay_buffer,
                sampler=sampler,
                steps_per_epoch=20,
                target_update_tau=1e-2,
                n_train_steps=50,
                discount=0.9,
                policy_weight_decay=0.01,
                qf_weight_decay=0.01,
                min_buffer_size=int(5e3),
                exploration_policy=exploration_policy,
            )
            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
            assert last_avg_ret > 10

            env.close()

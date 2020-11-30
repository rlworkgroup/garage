"""Create a test that fails when garage.tf.algos.TD3 performance is too low."""

import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import TD3
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTD3(TfGraphTestCase):
    """Tests for TD3 algo."""

    @pytest.mark.mujoco_long
    def test_td3_pendulum(self):
        """Test TD3 with Pendulum environment."""
        with TFTrainer(snapshot_config) as trainer:
            n_epochs = 10
            steps_per_epoch = 20
            sampler_batch_size = 250
            num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

            env = GymEnv('InvertedDoublePendulum-v2', max_episode_length=100)

            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[400, 300],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)

            exploration_policy = AddGaussianNoise(
                env.spec,
                policy,
                total_timesteps=num_timesteps,
                max_sigma=0.1,
                min_sigma=0.1)

            qf = ContinuousMLPQFunction(name='ContinuousMLPQFunction',
                                        env_spec=env.spec,
                                        hidden_sizes=[400, 300],
                                        action_merge_layer=0,
                                        hidden_nonlinearity=tf.nn.relu)

            qf2 = ContinuousMLPQFunction(name='ContinuousMLPQFunction2',
                                         env_spec=env.spec,
                                         hidden_sizes=[400, 300],
                                         action_merge_layer=0,
                                         hidden_nonlinearity=tf.nn.relu)

            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

            sampler = LocalSampler(
                agents=exploration_policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True,
                worker_class=FragmentWorker)

            algo = TD3(env_spec=env.spec,
                       policy=policy,
                       policy_lr=1e-3,
                       qf_lr=1e-3,
                       qf=qf,
                       qf2=qf2,
                       replay_buffer=replay_buffer,
                       sampler=sampler,
                       steps_per_epoch=steps_per_epoch,
                       target_update_tau=0.005,
                       n_train_steps=50,
                       discount=0.99,
                       min_buffer_size=int(1e4),
                       buffer_batch_size=100,
                       policy_weight_decay=0.001,
                       qf_weight_decay=0.001,
                       exploration_policy=exploration_policy,
                       policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                       qf_optimizer=tf.compat.v1.train.AdamOptimizer)

            trainer.setup(algo, env)
            last_avg_ret = trainer.train(n_epochs=n_epochs,
                                         batch_size=sampler_batch_size)
            assert last_avg_ret > 200

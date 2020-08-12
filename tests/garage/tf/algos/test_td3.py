"""Create a test that fails when garage.tf.algos.TD3 performance is too low."""

import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.tf.algos import TD3
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTD3(TfGraphTestCase):
    """Tests for TD3 algo."""

    @pytest.mark.mujoco_long
    def test_td3_pendulum(self):
        """Test TD3 with Pendulum environment."""
        with LocalTFRunner(snapshot_config) as runner:
            env = GymEnv('InvertedDoublePendulum-v2')

            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[400, 300],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)

            exploration_policy = AddGaussianNoise(env.spec,
                                                  policy,
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

            algo = TD3(env_spec=env.spec,
                       policy=policy,
                       policy_lr=1e-3,
                       qf_lr=1e-3,
                       qf=qf,
                       qf2=qf2,
                       replay_buffer=replay_buffer,
                       max_episode_length=100,
                       steps_per_epoch=20,
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

            runner.setup(algo, env, sampler_cls=LocalSampler)
            last_avg_ret = runner.train(n_epochs=10, batch_size=250)
            assert last_avg_ret > 200

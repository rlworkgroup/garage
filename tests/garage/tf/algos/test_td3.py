"""Create a test that fails when garage.tf.algos.TD3 performance is too low."""

import gym
import tensorflow as tf

from garage.np.exploration_strategies.gaussian_strategy import GaussianStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTD3(TfGraphTestCase):
    """Tests for TD3 algo."""

    def test_td3_pendulum(self):
        """Test TD3 with Pendulum environment."""
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

            action_noise = GaussianStrategy(env.spec,
                                            max_sigma=0.1,
                                            min_sigma=0.1)

            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[400, 300],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)

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

            replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                               size_in_transitions=int(1e6),
                                               time_horizon=250)

            algo = TD3(env_spec=env.spec,
                       policy=policy,
                       policy_lr=1e-3,
                       qf_lr=1e-3,
                       qf=qf,
                       qf2=qf2,
                       replay_buffer=replay_buffer,
                       target_update_tau=0.005,
                       n_epoch_cycles=20,
                       n_train_steps=50,
                       discount=0.99,
                       smooth_return=False,
                       min_buffer_size=int(1e4),
                       buffer_batch_size=100,
                       policy_weight_decay=0.001,
                       qf_weight_decay=0.001,
                       exploration_strategy=action_noise,
                       policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                       qf_optimizer=tf.compat.v1.train.AdamOptimizer)

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10,
                                        n_epoch_cycles=20,
                                        batch_size=250)
            assert last_avg_ret > 400

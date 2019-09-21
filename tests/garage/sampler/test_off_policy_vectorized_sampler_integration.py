import gym
import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.sampler import OffPolicyVectorizedSampler
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config, TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDictEnv
from tests.fixtures.policies import DummyPolicy
from tests.fixtures.tf.algos.dummy_off_policy_algo import DummyOffPolicyAlgo


class TestOffPolicyVectorizedSampler(TfGraphTestCase):

    def test_no_reset(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            # This tests if off-policy sampler respect batch_size
            # when no_reset is set to True
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            action_noise = OUStrategy(env.spec, sigma=0.2)
            policy = ContinuousMLPPolicy(env_spec=env.spec,
                                         hidden_sizes=[64, 64],
                                         hidden_nonlinearity=tf.nn.relu,
                                         output_nonlinearity=tf.nn.tanh)
            qf = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=tf.nn.relu)
            replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                               size_in_transitions=int(1e6),
                                               time_horizon=100)
            algo = DDPG(
                env_spec=env.spec,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                replay_buffer=replay_buffer,
                target_update_tau=1e-2,
                n_train_steps=50,
                discount=0.9,
                min_buffer_size=int(1e4),
                exploration_strategy=action_noise,
            )

            sampler = OffPolicyVectorizedSampler(algo, env, 1, no_reset=True)
            sampler.start_worker()

            runner.initialize_tf_vars()

            paths1 = sampler.obtain_samples(0, 5)
            paths2 = sampler.obtain_samples(0, 5)

            len1 = sum([len(path['rewards']) for path in paths1])
            len2 = sum([len(path['rewards']) for path in paths2])

            assert len1 == 5 and len2 == 5, 'Sampler should respect batch_size'
            # yapf: disable
            # When done is False in 1st sampling, the next sampling should be
            # stacked with the last batch in 1st sampling
            case1 = (len(paths1[-1]['rewards']) + len(paths2[0]['rewards'])
                     == paths2[0]['running_length'])
            # When done is True in 1st sampling, the next sampling should be
            # separated
            case2 = len(paths2[0]['rewards']) == paths2[0]['running_length']
            done = paths1[-1]['dones'][-1]
            assert (
                (not done and case1) or (done and case2)
            ), 'Running length should be the length of full path'

            # yapf: enable
            case1 = np.isclose(
                paths1[-1]['rewards'].sum() + paths2[0]['rewards'].sum(),
                paths2[0]['undiscounted_return'])
            case2 = np.isclose(paths2[0]['rewards'].sum(),
                               paths2[0]['undiscounted_return'])
            assert (
                (not done and case1) or (done and case2)
            ), 'Undiscounted_return should be the sum of rewards of full path'

    def test_algo_with_goal_without_es(self):
        # This tests if sampler works properly when algorithm
        # includes goal but is without exploration policy
        env = DummyDictEnv()
        policy = DummyPolicy(env)
        replay_buffer = SimpleReplayBuffer(env_spec=env,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)
        algo = DummyOffPolicyAlgo(env_spec=env,
                                  qf=None,
                                  replay_buffer=replay_buffer,
                                  policy=policy,
                                  exploration_strategy=None,
                                  input_include_goal=True)

        sampler = OffPolicyVectorizedSampler(algo, env, 1, no_reset=True)
        sampler.start_worker()
        sampler.obtain_samples(0, 30)

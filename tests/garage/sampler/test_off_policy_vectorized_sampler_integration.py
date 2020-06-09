import gym
import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import OffPolicyVectorizedSampler
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config, TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDictEnv
from tests.fixtures.policies import DummyPolicy
from tests.fixtures.tf.algos.dummy_off_policy_algo import DummyOffPolicyAlgo


class TestOffPolicyVectorizedSampler(TfGraphTestCase):

    @pytest.mark.mujoco
    def test_no_reset(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            # This tests if off-policy sampler respect batch_size
            # when no_reset is set to True
            env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
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
                exploration_policy=exploration_policy,
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

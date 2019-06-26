import pickle

import gym
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalLSTMPolicyWithModel
from tests.fixtures import TfGraphTestCase


class TestFiniteDifferenceHvp(TfGraphTestCase):
    def test_is_pickleable(self):
        f = FiniteDifferenceHvp(base_eps=2)

        with LocalRunner(self.sess) as runner:
            env = TfEnv(normalize(gym.make('CartPole-v1')))

            policy = CategoricalLSTMPolicyWithModel(
                name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                max_kl_step=0.01,
                optimizer_args=dict(hvp_approach=f))

            runner.setup(algo, env)

        h = pickle.dumps(f)
        with tf.Session(graph=tf.Graph()):
            f_pickled = pickle.loads(h)

        assert f_pickled.base_eps == f.base_eps

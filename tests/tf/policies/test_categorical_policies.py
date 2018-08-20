"""
This script creates a unittest that tests Categorical policies in
garage.tf.policies.
"""
import unittest

import gym
from nose2 import tools
import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalGRUPolicy
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalMLPPolicy

policies = [CategoricalGRUPolicy, CategoricalLSTMPolicy, CategoricalMLPPolicy]


class TestCategoricalPolicies(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session(graph=tf.Graph())
        self.sess.__enter__()
        logger._tensorboard = TensorBoardOutput()

    @tools.params(*policies)
    def test_categorical_policies(self, policy_cls):
        env = TfEnv(normalize(gym.make("CartPole-v0")))

        policy = policy_cls(name="policy", env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
            plot=True,
            optimizer=ConjugateGradientOptimizer,
            optimizer_args=dict(
                hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        )
        algo.train(sess=self.sess)

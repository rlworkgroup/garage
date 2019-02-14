"""
This script creates a test that fails when garage.tf.algos.TRPO performance is
too low.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
import garage.misc.logger as logger
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestTRPO(TfGraphTestCase):
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        logger.reset()
        env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2048,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            gae_lambda=0.98,
            policy_ent_coeff=0.0,
            plot=False,
        )
        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 50

        env.close()

    def test_trpo_unknown_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        logger.reset()
        env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(NotImplementedError) as context:
            TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=2048,
                max_path_length=100,
                n_itr=10,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0,
                plot=False,
                kl_constraint="random kl_constraint",
            )
        assert "Unknown KLConstraint" in str(context.exception)

        env.close()

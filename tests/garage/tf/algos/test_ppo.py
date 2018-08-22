"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import GarageTestCase


class TestPPO(GarageTestCase):
    def test_ppo_pendulum(self):
        """Test PPO with Pendulum environment."""
        logger._tensorboard = TensorBoardOutput()
        env = TfEnv(normalize(gym.make("Pendulum-v0")))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1024,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            gae_lambda=0.98,
            policy_ent_coeff=0.0,
            plot=False,
        )
        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > -1000

"""
This script creates a test that fails when garage.tf.algos.VPG performance is
too low.
"""
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestVPG(TfGraphTestCase):
    def test_vpg_cartpole(self):
        """Test VPG with Cartpole environment."""
        logger._tensorboard = TensorBoardOutput()
        env = TfEnv(normalize(CartpoleEnv()))

        policy = GaussianMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=10000,
            max_path_length=100,
            n_itr=20,
            discount=0.99,
            optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 195

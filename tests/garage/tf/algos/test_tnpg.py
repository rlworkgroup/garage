import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.tf.algos import TNPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestTNPG(TfGraphTestCase):
    def test_tnpg_inverted_pendulum(self):
        """Test TNPG with InvertedPendulum-v2 environment."""
        env = TfEnv(normalize(gym.make("InvertedPendulum-v2")))

        policy = GaussianMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TNPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=10000,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            optimizer_args=dict(reg_coeff=5e-1))

        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 30

        env.close()

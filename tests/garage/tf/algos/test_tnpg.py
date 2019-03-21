import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.tf.algos import TNPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestTNPG(TfGraphTestCase):
    def test_tnpg_inverted_pendulum(self):
        """Test TNPG with InvertedPendulum-v2 environment."""
        with LocalRunner(self.sess) as runner:
            env = TfEnv(normalize(gym.make("InvertedPendulum-v2")))

            policy = GaussianMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TNPG(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                optimizer_args=dict(reg_coeff=5e-1))

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 30

            env.close()

from garage.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestVPG(TfGraphTestCase):
    def test_vpg_cartpole(self):
        """Test VPG with CartPole-v1 environment."""
        env = TfEnv(env_name="CartPole-v1")

        policy = CategoricalMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=10000,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 90

        env.close()

"""
This script creates a test that fails when garage.tf.algos.TRPO performance is
too low.
"""
import gym

from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalLSTMPolicyWithModel
from tests.fixtures import TfGraphTestCase


class TestTRPO(TfGraphTestCase):
    def test_trpo_recurrent_cartpole(self):
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
                optimizer_args=dict(
                    hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)))

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 90

            env.close()

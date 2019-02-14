from dm_control.suite import ALL_TASKS

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestDmControlTfPolicy(TfGraphTestCase):
    def test_dm_control_tf_policy(self):
        task = ALL_TASKS[0]

        with self.graph.as_default():
            env = TfEnv(DmControlEnv.from_suite(*task))

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=10,
                max_path_length=5,
                n_itr=1,
                discount=0.99,
                step_size=0.01,
            )
            algo.train()
            env.close()

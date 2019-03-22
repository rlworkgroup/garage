from dm_control.suite import ALL_TASKS

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.experiment import LocalRunner
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestDmControlTfPolicy(TfGraphTestCase):
    def test_dm_control_tf_policy(self):
        task = ALL_TASKS[0]

        env = TfEnv(DmControlEnv.from_suite(*task))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            batch_size=10,
            max_path_length=5,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
        )

        runner = LocalRunner(self.sess)
        runner.setup(algo, env)
        runner.train(n_epochs=1, batch_size=10)

        env.close()

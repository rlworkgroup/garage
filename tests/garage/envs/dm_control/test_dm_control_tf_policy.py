from dm_control.suite import ALL_TASKS
import pytest

from garage.envs.dm_control import DMControlEnv
from garage.experiment import LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy

from tests.fixtures import snapshot_config, TfGraphTestCase


@pytest.mark.mujoco
class TestDmControlTfPolicy(TfGraphTestCase):

    def test_dm_control_tf_policy(self):
        task = ALL_TASKS[0]

        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = DMControlEnv.from_suite(*task)

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_episode_length=5,
                discount=0.99,
                max_kl_step=0.01,
            )

            runner.setup(algo, env, sampler_cls=LocalSampler)
            runner.train(n_epochs=1, batch_size=10)

            env.close()

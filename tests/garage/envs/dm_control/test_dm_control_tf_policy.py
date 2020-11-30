from dm_control.suite import ALL_TASKS
import pytest

from garage.envs.dm_control import DMControlEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


@pytest.mark.mujoco
class TestDmControlTfPolicy(TfGraphTestCase):

    def test_dm_control_tf_policy(self):
        task = ALL_TASKS[0]

        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = DMControlEnv.from_suite(*task)

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                sampler=sampler,
                discount=0.99,
                max_kl_step=0.01,
            )

            trainer.setup(algo, env)
            trainer.train(n_epochs=1, batch_size=10)

            env.close()

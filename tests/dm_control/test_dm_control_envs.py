import pickle

from dm_control.suite import ALL_TASKS
from nose2 import tools

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy
from garage.tf.spaces import Box
from tests.fixtures import GarageTestCase

dm_control_envs = []
for task in ALL_TASKS:
    dm_control_envs.append(
        TfEnv(DmControlEnv(domain_name=task[0], task_name=task[1])))


class TestDmControlEnvs(GarageTestCase):
    @tools.params(*dm_control_envs)
    def test_dm_control_envs(self, env):
        # Test pickling
        pickle.loads(pickle.dumps(env))

        if isinstance(env.spec.action_space, Box):
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
            )
        else:
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
            )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=100,
            max_path_length=10,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
        )
        algo.train()

import gc

from dm_control.suite import ALL_TASKS
from nose2 import tools

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy
from garage.tf.spaces import Box
from tests.fixtures import GarageTestCase

all_tasks_list = []
for task in ALL_TASKS[0:1]:
    all_tasks_list.append([task[0], task[1]])


class TestDmControlPolicy(GarageTestCase):
    @tools.params(*all_tasks_list)
    def test_dm_control_policy(self, task):
        env = TfEnv(DmControlEnv(domain_name=task[0], task_name=task[1]))

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
            batch_size=10,
            max_path_length=5,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
        )
        print("Testing", env.__class__)
        algo.train()
        env.close()

        # Needed b/c parameterized test causes memory leaks
        del algo
        del baseline
        del policy
        del env
        del task
        gc.collect()

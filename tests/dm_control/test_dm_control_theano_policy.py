import unittest

from dm_control.suite import ALL_TASKS

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


class TestDmControlTheanoPolicy(unittest.TestCase):
    def test_dm_control_theano_policy(self):
        task = ALL_TASKS[0]

        env = TheanoEnv(DmControlEnv(domain_name=task[0], task_name=task[1]))

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

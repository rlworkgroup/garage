import gc
import unittest

from dm_control.suite import ALL_TASKS
import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.envs.dm_control import DmControlEnv
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.spaces import Box


class TestDmControlPolicies(unittest.TestCase):
    def test_dm_control_policies(self):
        for task in ALL_TASKS[::2]:
            g = tf.Graph()
            with g.as_default():
                env = TfEnv(
                    DmControlEnv(domain_name=task[0], task_name=task[1]))

                if isinstance(env.spec.action_space, Box):
                    policy = GaussianMLPPolicy(
                        env_spec=env.spec,
                        hidden_sizes=(32, 32),
                    )
                # ValueError: Tensor("CategoricalMLPPolicy/Actions:0",
                # shape=(), dtype=string) must be from the same graph
                # as Tensor("GaussianMLPPolicy/Actions:0", shape=(),
                # dtype=string).
                else:
                    continue

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

            del g
            gc.collect()

import os
import signal
import time
import unittest

import psutil

from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy

def fixture_exp():
    with LocalRunner() as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(
            name='policy', env_spec=env.spec, hidden_sizes=(8, 8))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            optimizer_args=dict(
                tf_optimizer_args=dict(learning_rate=0.01, )))

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=100)


class TestResume(unittest.TestCase):
    def test_resume(self):
        newpid = os.fork()
        if newpid == 0:
            fixture_exp()
        else:
            time.sleep(5)
            psutil.Process(newpid).terminal()
            self.assertIsNone(psutil.pid_exists(newpid))

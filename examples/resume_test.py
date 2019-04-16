# from garage.experiment.local_tf_runner import LocalRunner
#
# with LocalRunner() as runner:
#     runner.restore('/home/kzhu/garage/data/local/experiment/experiment_2019_03_28_15_13_27_0001')
#     runner.train(100, batch_size=2000)
from garage.experiment import run_experiment

import os
import signal
import tempfile
import time

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.logger import logger, snapshotter

def run_task(snapshot_dir):
    with LocalRunner() as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(
            name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)

def restore_task(snapshot_dir):
    pass

if __name__ == '__main__':
    snapshot_dir = tempfile.TemporaryDirectory()
    snapshotter.snapshot_dir = snapshot_dir
    snapshotter.snapshot_mode = 'last'

    newpid = os.fork()
    if newpid == 0:
        run_task(snapshot_dir)
    else:
        time.sleep(10)
        os.kill(newpid, signal.SIGTERM)
        restore_task(snapshot_dir)

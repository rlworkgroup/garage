"""
This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5
"""

import gym

from garage.baselines import LinearFeatureBaseline
from garage.experiment import run_experiment
from garage.tf.algos import REPS
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy


def run_task(*_):
    """Wrap REPS training task in the run_task function."""
    env = TfEnv(gym.make("CartPole-v0"))

    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = REPS(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        plot=False)

    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)

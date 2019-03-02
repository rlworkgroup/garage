"""
This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5
"""

import gym

from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.tf.algos import REPS
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy

with LocalRunner() as runner:
    env = TfEnv(gym.make("CartPole-v0"))

    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = REPS(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=100,
        discount=0.99)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=4000, plot=False)

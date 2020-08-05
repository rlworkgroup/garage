#!/usr/bin/env python3
"""This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5

"""

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import REPS
from garage.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def reps_gym_cartpole(ctxt=None, seed=1):
    """Train REPS with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GymEnv('CartPole-v0')

        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = REPS(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_episode_length=100,
                    discount=0.99)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000, plot=False)


reps_gym_cartpole()

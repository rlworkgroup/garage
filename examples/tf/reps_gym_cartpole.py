#!/usr/bin/env python3
"""This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5

"""

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import REPS
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def reps_gym_cartpole(ctxt=None, seed=1):
    """Train REPS with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('CartPole-v0')

        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = REPS(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000, plot=False)


reps_gym_cartpole()

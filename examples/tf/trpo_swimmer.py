#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def trpo_swimmer(ctxt=None, seed=1, batch_size=4000):
    """Train TRPO with Swimmer-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        batch_size (int): Number of timesteps to use in each training step.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = GymEnv('Swimmer-v2')

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_episode_length=500,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=40, batch_size=batch_size)


trpo_swimmer()

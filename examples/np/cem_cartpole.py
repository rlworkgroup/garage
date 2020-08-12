#!/usr/bin/env python3
"""This is an example to train a task with Cross Entropy Method.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.algos import CEM
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def cem_cartpole(ctxt=None, seed=1):
    """Train CEM with Cartpole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GymEnv('CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 20

        algo = CEM(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   best_frac=0.05,
                   max_episode_length=100,
                   n_samples=n_samples)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=1000)


cem_cartpole(seed=1)

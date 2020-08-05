#!/usr/bin/env python3
"""This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 16
"""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def vpg_cartpole(ctxt=None, seed=1):
    """Train VPG with CartPole-v1 environment.

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

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_episode_length=100,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=10000)


vpg_cartpole()

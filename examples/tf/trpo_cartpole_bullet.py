#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Here it runs CartPoleBulletEnv environment with 100 iterations.

"""
import gym

from garage import wrap_experiment
from garage.envs.bullet import BulletEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def trpo_cartpole_bullet(ctxt=None, seed=1):
    """Train TRPO with Pybullet's CartPoleBulletEnv environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = BulletEnv(
            gym.make('CartPoleBulletEnv-v1',
                     renders=False,
                     discrete_actions=True))

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_episode_length=1000,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


trpo_cartpole_bullet()

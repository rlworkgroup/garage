#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Here it runs CartPoleBulletEnv environment with 100 iterations.

"""
import gym

from garage import wrap_experiment
from garage.envs.bullet import BulletEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def trpo_cartpole_bullet(ctxt=None, seed=1):
    """Train TRPO with Pybullet's CartPoleBulletEnv environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = BulletEnv(
            gym.make('CartPoleBulletEnv-v1',
                     renders=False,
                     discrete_actions=True))

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000)


trpo_cartpole_bullet()

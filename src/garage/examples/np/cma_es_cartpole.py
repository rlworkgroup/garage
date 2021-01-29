#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def cma_es_cartpole(ctxt=None, seed=1):
    """Train CMA_ES with Cartpole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = GymEnv('CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        n_samples = 20

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)

        algo = CMAES(env_spec=env.spec,
                     policy=policy,
                     sampler=sampler,
                     n_samples=n_samples)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=1000)


cma_es_cartpole()

#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def trpo_gym_tf_cartpole(ctxt=None, seed=1):
    """Train TRPO with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('CartPole-v0')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            sampler=sampler,
            discount=0.99,
            max_kl_step=0.01,
        )

        trainer.setup(algo, env)
        trainer.train(n_epochs=120, batch_size=4000)


trpo_gym_tf_cartpole()

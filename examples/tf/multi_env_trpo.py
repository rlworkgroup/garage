#!/usr/bin/env python3
"""This is an example to train multiple tasks with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import normalize, PointEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def multi_env_trpo(ctxt=None, seed=1):
    """Train TRPO on two different PointEnv instances.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env1 = normalize(PointEnv(goal=(-1., 0.), max_episode_length=100))
        env2 = normalize(PointEnv(goal=(1., 0.), max_episode_length=100))
        env = MultiEnvWrapper([env1, env2])

        policy = GaussianMLPPolicy(env_spec=env.spec)

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
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    policy_ent_coeff=0.0)

        trainer.setup(algo, env)
        trainer.train(n_epochs=40, batch_size=2048, plot=False)


multi_env_trpo()

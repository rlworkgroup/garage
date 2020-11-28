#!/usr/bin/env python3
"""This is an example to train multiple tasks with PPO algorithm."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import PPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def multi_env_ppo(ctxt=None, seed=1):
    """Train PPO on two Atari environments simultaneously.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env1 = normalize(GymEnv('Adventure-ram-v4'))
        env2 = normalize(GymEnv('Alien-ram-v4'))
        env = MultiEnvWrapper([env1, env2])

        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_ent_coeff=0.0,
                   optimizer_args=dict(
                       batch_size=32,
                       max_optimization_epochs=10,
                       learning_rate=1e-3,
                   ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=120, batch_size=2048, plot=False)


multi_env_ppo()

"""Benchmarking experiment of the CategoricalGRUPolicy."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.policies import CategoricalGRUPolicy


@wrap_experiment
def categorical_gru_policy(ctxt, env_id, seed):
    """Create Categorical CNN Policy on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt) as runner:
        env = normalize(GarageEnv(gym.make(env_id)))

        policy = CategoricalGRUPolicy(
            env_spec=env.spec,
            hidden_dim=32,
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_episode_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_episode_length=10,
                learning_rate=1e-3,
            ),
        )

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=488, batch_size=2048)

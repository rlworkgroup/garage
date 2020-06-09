"""Benchmarking experiment of the GaussianGRUPolicy."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianGRUPolicy


@wrap_experiment
def gaussian_gru_policy(ctxt, env_id, seed):
    """Create Gaussian GRU Policy on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt) as runner:
        env = GarageEnv(normalize(gym.make(env_id)))

        policy = GaussianGRUPolicy(
            env_spec=env.spec,
            hidden_dim=32,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    learning_rate=1e-3,
                ),
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                learning_rate=1e-3,
            ),
        )

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=5, batch_size=2048)

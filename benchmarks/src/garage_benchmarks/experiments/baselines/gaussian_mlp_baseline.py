"""Benchmarking experiment of the GaussianMLPBaseline."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.sampler import RaySampler
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def gaussian_mlp_baseline(ctxt, env_id, seed):
    """Create Gaussian MLP Baseline on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        env = normalize(GymEnv(env_id))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            use_trust_region=False,
            optimizer=FirstOrderOptimizer,
            optimizer_args=dict(
                batch_size=32,
                max_optimization_epochs=10,
                learning_rate=1e-3,
            ),
        )

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = PPO(
            env_spec=env.spec,
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
            ),
        )
        trainer.setup(algo, env)
        trainer.train(n_epochs=5, batch_size=2048)

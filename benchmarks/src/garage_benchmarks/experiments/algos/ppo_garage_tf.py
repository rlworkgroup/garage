"""A regression test for automatic benchmarking garage-TensorFlow-PPO."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.sampler import RaySampler
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline as TF_GMB
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.trainer import TFTrainer

hyper_parameters = {
    'n_epochs': 500,
    'batch_size': 1024,
}


@wrap_experiment
def ppo_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow PPO model and training.

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

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = TF_GMB(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            use_trust_region=False,
            optimizer=FirstOrderOptimizer,
            optimizer_args=dict(
                batch_size=32,
                max_optimization_epochs=10,
                learning_rate=3e-4,
            ),
        )

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TF_PPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=0.99,
                      gae_lambda=0.95,
                      center_adv=True,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32,
                                          max_optimization_epochs=10,
                                          learning_rate=3e-4,
                                          verbose=True))

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_parameters['n_epochs'],
                      batch_size=hyper_parameters['batch_size'])

"""A regression test for automatic benchmarking garage-TensorFlow-PPO."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.tf.algos import PPO as TF_PPO
from garage.tf.baselines import GaussianMLPBaseline as TF_GMB
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy as TF_GMP

hyper_parameters = {
    'n_epochs': 500,
    'max_path_length': 100,
    'batch_size': 1024,
}


@wrap_experiment
def ppo_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow PPO model and training.

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

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = TF_GMB(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    learning_rate=3e-4,
                ),
            ),
        )

        algo = TF_PPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=hyper_parameters['max_path_length'],
                      discount=0.99,
                      gae_lambda=0.95,
                      center_adv=True,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32,
                                          max_epochs=10,
                                          learning_rate=3e-4,
                                          verbose=True))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

"""A regression test for automatic benchmarking garage-TensorFlow-VPG."""
import gym
import tensorflow as tf

from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalTFRunner
from garage.tf.algos import A2C as TF_A2C
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage import wrap_experiment

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'policy_ent_coeff': 0.02,
    'max_episode_length': 100,
    'batch_size': 10000,
}


@wrap_experiment
def a2c_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow A2C model and training.

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
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            use_trust_region=True,
        )

        algo = TF_A2C(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            policy_ent_coeff=hyper_parameters['policy_ent_coeff'],
            stop_entropy_gradient=True,
            max_episode_length=hyper_parameters['max_episode_length'],
            discount=hyper_parameters['discount'],
            optimizer_args=dict(
                learning_rate=hyper_parameters['learning_rate'], ))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

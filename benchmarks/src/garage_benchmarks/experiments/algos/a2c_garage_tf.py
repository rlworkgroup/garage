"""A regression test for automatic benchmarking garage-TensorFlow-VPG."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import A2C as TF_A2C
from garage.tf.policies import GaussianMLPPolicy as TF_GMP

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'learning_rate': 1e-3,
    'discount': 0.99,
    'n_epochs': 2000,
    'batch_size': 512,
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
        env = normalize(GymEnv(env_id))

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TF_A2C(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_episode_length=env.spec.max_episode_length,
                      discount=hyper_parameters['discount'],
                      stop_entropy_gradient=True,
                      center_adv=False,
                      optimizer_args=dict(
                          learning_rate=hyper_parameters['learning_rate'], ))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

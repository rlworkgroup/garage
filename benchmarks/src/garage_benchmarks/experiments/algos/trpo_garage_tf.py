"""A regression test for automatic benchmarking garage-TensorFlow-TRPO."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'n_epochs': 999,
    'batch_size': 1024,
}


@wrap_experiment
def trpo_garage_tf(ctxt, env_id, seed):
    """Create garage Tensorflow TROI model and training.

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
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    max_kl_step=hyper_parameters['max_kl'])

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_parameters['n_epochs'],
                      batch_size=hyper_parameters['batch_size'])

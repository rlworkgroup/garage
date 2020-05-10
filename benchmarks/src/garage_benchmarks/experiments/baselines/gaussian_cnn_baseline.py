"""Benchmarking experiment of the GaussianCNNBaseline."""
import gym

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalCNNPolicy

params = {
    'conv_filters': (32, 64, 64),
    'conv_filter_sizes': (5, 3, 2),
    'conv_strides': (4, 2, 1),
    'conv_pad': 'VALID',
    'hidden_sizes': (256, ),
    'n_epochs': 1000,
    'batch_size': 2048,
    'use_trust_region': True
}


@wrap_experiment
def gaussian_cnn_baseline(ctxt, env_id, seed):
    """Create Gaussian CNN Baseline on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt, max_cpus=12) as runner:
        env = TfEnv(normalize(gym.make(env_id)))

        policy = CategoricalCNNPolicy(
            env_spec=env.spec,
            conv_filters=params['conv_filters'],
            conv_filter_sizes=params['conv_filter_sizes'],
            conv_strides=params['conv_strides'],
            conv_pad=params['conv_pad'],
            hidden_sizes=params['hidden_sizes'])

        baseline = GaussianCNNBaseline(
            env_spec=env.spec,
            regressor_args=dict(num_filters=params['conv_filters'],
                                filter_dims=params['conv_filter_sizes'],
                                strides=params['conv_strides'],
                                padding=params['conv_pad'],
                                hidden_sizes=params['hidden_sizes'],
                                use_trust_region=params['use_trust_region']))

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            flatten_input=False,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(learning_rate=1e-3),
            ),
        )

        runner.setup(algo, env)
        runner.train(n_epochs=params['n_epochs'],
                     batch_size=params['batch_size'])

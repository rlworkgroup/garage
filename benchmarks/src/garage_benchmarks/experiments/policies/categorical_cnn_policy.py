"""Benchmarking experiment of the CategoricalCNNPolicy."""
import gym

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.policies import CategoricalCNNPolicy

hyper_params = {
    'conv_filters': (
                        (32, (5, 5)),
                        (64, (3, 3)),
                        (64, (2, 2)),
                    ),
    'conv_strides': (4, 2, 1),
    'conv_pad': 'VALID',
    'hidden_sizes': (256, ),
    'n_epochs': 3,
    'batch_size': 2048,
    'use_trust_region': True
}  # yapf: disable


@wrap_experiment
def categorical_cnn_policy(ctxt, env_id, seed):
    """Create Categorical CNN Policy on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt, max_cpus=12) as runner:
        env = GarageEnv(normalize(gym.make(env_id)))

        policy = CategoricalCNNPolicy(
            env_spec=env.spec,
            conv_filters=hyper_params['conv_filters'],
            conv_strides=hyper_params['conv_strides'],
            conv_pad=hyper_params['conv_pad'],
            hidden_sizes=hyper_params['hidden_sizes'])

        baseline = GaussianCNNBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                filters=hyper_params['conv_filters'],
                strides=hyper_params['conv_strides'],
                padding=hyper_params['conv_pad'],
                hidden_sizes=hyper_params['hidden_sizes'],
                use_trust_region=hyper_params['use_trust_region']))

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
            flatten_input=False,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_params['n_epochs'],
                     batch_size=hyper_params['batch_size'])

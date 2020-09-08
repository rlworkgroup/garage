"""Benchmarking experiment of the CategoricalCNNPolicy."""
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.policies import CategoricalCNNPolicy
from garage.trainer import TFTrainer

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
            configuration used by Trainer to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        env = normalize(GymEnv(env_id))

        policy = CategoricalCNNPolicy(
            env_spec=env.spec,
            filters=hyper_params['conv_filters'],
            strides=hyper_params['conv_strides'],
            padding=hyper_params['conv_pad'],
            hidden_sizes=hyper_params['hidden_sizes'])

        baseline = GaussianCNNBaseline(
            env_spec=env.spec,
            filters=hyper_params['conv_filters'],
            strides=hyper_params['conv_strides'],
            padding=hyper_params['conv_pad'],
            hidden_sizes=hyper_params['hidden_sizes'],
            use_trust_region=hyper_params['use_trust_region'])

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_ent_coeff=0.0,
                   optimizer_args=dict(
                       batch_size=32,
                       max_optimization_epochs=10,
                       learning_rate=1e-3,
                   ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_params['n_epochs'],
                      batch_size=hyper_params['batch_size'])

"""Benchmarking experiment of the ContinuousMLPBaseline."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.tf.algos import PPO
from garage.tf.baselines import ContinuousMLPBaseline
from garage.tf.policies import GaussianLSTMPolicy

hyper_params = {
    'policy_hidden_sizes': 32,
    'hidden_nonlinearity': tf.nn.tanh,
    'n_envs': 8,
    'n_epochs': 20,
    'n_rollout_steps': 2048,
    'discount': 0.99,
    'max_path_length': 100,
    'gae_lambda': 0.95,
    'lr_clip_range': 0.2,
    'policy_ent_coeff': 0.02,
    'entropy_method': 'max',
    'center_adv': False,
    'num_proc': 8
}


@wrap_experiment
def continuous_mlp_baseline(ctxt, env_id, seed):
    """Create Continuous MLP Baseline on TF-PPO.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt, max_cpus=hyper_params['num_proc']) as runner:
        env = GarageEnv(normalize(gym.make(env_id)))

        policy = GaussianLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=hyper_params['policy_hidden_sizes'],
            hidden_nonlinearity=hyper_params['hidden_nonlinearity'],
        )

        baseline = ContinuousMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(64, 64)),
        )

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=hyper_params['max_path_length'],
                   discount=hyper_params['discount'],
                   gae_lambda=hyper_params['gae_lambda'],
                   lr_clip_range=hyper_params['lr_clip_range'],
                   entropy_method=hyper_params['entropy_method'],
                   policy_ent_coeff=hyper_params['policy_ent_coeff'],
                   optimizer_args=dict(
                       batch_size=32,
                       max_epochs=10,
                       learning_rate=1e-3,
                   ),
                   center_adv=hyper_params['center_adv'],
                   stop_entropy_gradient=True)

        runner.setup(algo,
                     env,
                     sampler_args=dict(n_envs=hyper_params['n_envs']))
        runner.train(n_epochs=hyper_params['n_epochs'],
                     batch_size=hyper_params['n_rollout_steps'])

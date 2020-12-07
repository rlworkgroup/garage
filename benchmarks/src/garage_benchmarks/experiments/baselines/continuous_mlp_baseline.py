"""Benchmarking experiment of the ContinuousMLPBaseline."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.sampler import RaySampler
from garage.tf.algos import PPO
from garage.tf.baselines import ContinuousMLPBaseline
from garage.tf.policies import GaussianLSTMPolicy
from garage.trainer import TFTrainer

hyper_params = {
    'policy_hidden_sizes': 32,
    'hidden_nonlinearity': tf.nn.tanh,
    'n_epochs': 20,
    'n_exploration_steps': 2048,
    'discount': 0.99,
    'gae_lambda': 0.95,
    'lr_clip_range': 0.2,
    'policy_ent_coeff': 0.02,
    'entropy_method': 'max',
    'center_adv': False,
}


@wrap_experiment
def continuous_mlp_baseline(ctxt, env_id, seed):
    """Create Continuous MLP Baseline on TF-PPO.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        env = normalize(GymEnv(env_id))

        policy = GaussianLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=hyper_params['policy_hidden_sizes'],
            hidden_nonlinearity=hyper_params['hidden_nonlinearity'],
        )

        baseline = ContinuousMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
        )

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=hyper_params['discount'],
                   gae_lambda=hyper_params['gae_lambda'],
                   lr_clip_range=hyper_params['lr_clip_range'],
                   entropy_method=hyper_params['entropy_method'],
                   policy_ent_coeff=hyper_params['policy_ent_coeff'],
                   optimizer_args=dict(
                       batch_size=32,
                       max_optimization_epochs=10,
                       learning_rate=1e-3,
                   ),
                   center_adv=hyper_params['center_adv'],
                   stop_entropy_gradient=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_params['n_epochs'],
                      batch_size=hyper_params['n_exploration_steps'])

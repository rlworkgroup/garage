"""Benchmarking experiment of the ContinuousMLPQFunction."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

hyper_params = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 300,
    'steps_per_epoch': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}


@wrap_experiment
def continuous_mlp_q_function(ctxt, env_id, seed):
    """Create Continuous MLP QFunction on TF-DDPG.

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

        action_noise = OUStrategy(env.spec, sigma=hyper_params['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name='ContinuousMLPPolicy',
            hidden_sizes=hyper_params['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=hyper_params['qf_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            name='ContinuousMLPQFunction')

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=hyper_params['replay_buffer_size'],
            time_horizon=hyper_params['n_rollout_steps'])

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    steps_per_epoch=hyper_params['steps_per_epoch'],
                    policy_lr=hyper_params['policy_lr'],
                    qf_lr=hyper_params['qf_lr'],
                    target_update_tau=hyper_params['tau'],
                    n_train_steps=hyper_params['n_train_steps'],
                    discount=hyper_params['discount'],
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        runner.setup(ddpg, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=hyper_params['n_epochs'],
                     batch_size=hyper_params['n_rollout_steps'])

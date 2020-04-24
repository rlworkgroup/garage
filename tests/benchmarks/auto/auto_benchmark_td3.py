"""A regression test over TD3 Algorithms for automatic benchmarking.
(garage-TensorFlow-TD3)
"""
import random

from baselines.bench import benchmarks
import gym
import pytest
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies.gaussian_strategy import GaussianStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests import benchmark_helper

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [400, 300],
    'qf_hidden_sizes': [400, 300],
    'n_epochs': 8,
    'n_trials': 6,
    'steps_per_epoch': 20,
    'n_rollout_steps': 250,
    'n_train_steps': 1,
    'discount': 0.99,
    'tau': 0.005,
    'replay_buffer_size': int(1e6),
    'sigma': 0.1,
    'smooth_return': False,
    'buffer_batch_size': 100,
    'min_buffer_size': int(1e4)
}

seeds = random.sample(range(100), hyper_parameters['n_trials'])
tasks = [
    task for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
    if task['env_id'] != 'Reacher-v2'
]


@pytest.mark.benchmark
def auto_benchmark_td3_garage_tf():
    """Create garage TensorFlow TD3 model and training.

    Training over different environments and seeds.

    """

    @wrap_experiment
    def td3_garage_tf(ctxt, env_id, seed):
        """Create garage TensorFlow TD3 model and training.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the
                snapshotter.
            env_id (str): Environment id of the task.
            seed (int): Random positive integer for the trial.

        """
        deterministic.set_seed(seed)

        with LocalTFRunner(ctxt) as runner:
            env = TfEnv(normalize(gym.make(env_id)))

            exploration_noise = GaussianStrategy(
                env.spec,
                max_sigma=hyper_parameters['sigma'],
                min_sigma=hyper_parameters['sigma'])

            policy = ContinuousMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=hyper_parameters['policy_hidden_sizes'],
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh)

            qf = ContinuousMLPQFunction(
                name='ContinuousMLPQFunction',
                env_spec=env.spec,
                hidden_sizes=hyper_parameters['qf_hidden_sizes'],
                action_merge_layer=0,
                hidden_nonlinearity=tf.nn.relu)

            qf2 = ContinuousMLPQFunction(
                name='ContinuousMLPQFunction2',
                env_spec=env.spec,
                hidden_sizes=hyper_parameters['qf_hidden_sizes'],
                action_merge_layer=0,
                hidden_nonlinearity=tf.nn.relu)

            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=hyper_parameters['replay_buffer_size'],
                time_horizon=hyper_parameters['n_rollout_steps'])

            td3 = TD3(env.spec,
                      policy=policy,
                      qf=qf,
                      qf2=qf2,
                      replay_buffer=replay_buffer,
                      steps_per_epoch=hyper_parameters['steps_per_epoch'],
                      policy_lr=hyper_parameters['policy_lr'],
                      qf_lr=hyper_parameters['qf_lr'],
                      target_update_tau=hyper_parameters['tau'],
                      n_train_steps=hyper_parameters['n_train_steps'],
                      discount=hyper_parameters['discount'],
                      smooth_return=hyper_parameters['smooth_return'],
                      min_buffer_size=hyper_parameters['min_buffer_size'],
                      buffer_batch_size=hyper_parameters['buffer_batch_size'],
                      exploration_strategy=exploration_noise,
                      policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                      qf_optimizer=tf.compat.v1.train.AdamOptimizer)

            runner.setup(td3, env)
            runner.train(n_epochs=hyper_parameters['n_epochs'],
                         batch_size=hyper_parameters['n_rollout_steps'])

    for env_id, seed, log_dir in benchmark_helper.iterate_experiments(
            td3_garage_tf.__name__, tasks, seeds):
        td3_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

"""A regression test for automatic benchmarking garage-TensorFlow-TD3."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.tf.algos import TD3
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [400, 300],
    'qf_hidden_sizes': [400, 300],
    'n_epochs': 8,
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
        env = GarageEnv(normalize(gym.make(env_id)))

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddGaussianNoise(
            env.spec,
            policy,
            max_sigma=hyper_parameters['sigma'],
            min_sigma=hyper_parameters['sigma'])

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

        replay_buffer = PathBuffer(
            capacity_in_transitions=hyper_parameters['replay_buffer_size'])

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
                  exploration_policy=exploration_policy,
                  policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                  qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        runner.setup(td3, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['n_rollout_steps'])

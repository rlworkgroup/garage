"""A regression test for automatic benchmarking garage-TensorFlow-HER."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import HerReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [256, 256, 256],
    'qf_hidden_sizes': [256, 256, 256],
    'n_epochs': 50,
    'steps_per_epoch': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 40,
    'discount': 0.9,
    'tau': 0.05,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
}


@wrap_experiment
def her_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow HER model and training.

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

        action_noise = OUStrategy(env.spec, sigma=hyper_parameters['sigma'])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['qf_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
        )

        replay_buffer = HerReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=hyper_parameters['replay_buffer_size'],
            time_horizon=hyper_parameters['n_rollout_steps'],
            replay_k=0.4,
            reward_fun=env.compute_reward,
        )

        algo = DDPG(
            env_spec=env.spec,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            steps_per_epoch=hyper_parameters['steps_per_epoch'],
            policy_lr=hyper_parameters['policy_lr'],
            qf_lr=hyper_parameters['qf_lr'],
            target_update_tau=hyper_parameters['tau'],
            n_train_steps=hyper_parameters['n_train_steps'],
            discount=hyper_parameters['discount'],
            exploration_strategy=action_noise,
            policy_optimizer=tf.compat.v1.train.AdamOptimizer,
            qf_optimizer=tf.compat.v1.train.AdamOptimizer,
            buffer_batch_size=256,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['n_rollout_steps'])

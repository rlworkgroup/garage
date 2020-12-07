"""A regression test for automatic benchmarking garage-TensorFlow-TD3."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import TD3
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [256, 256],
    'qf_hidden_sizes': [256, 256],
    'n_epochs': 250,
    'steps_per_epoch': 40,
    'n_exploration_steps': 100,
    'n_train_steps': 50,
    'discount': 0.99,
    'tau': 0.005,
    'replay_buffer_size': int(1e6),
    'sigma': 0.1,
    'buffer_batch_size': 100,
    'min_buffer_size': int(1e4)
}


@wrap_experiment
def td3_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow TD3 model and training.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        num_timesteps = (hyper_parameters['n_epochs'] *
                         hyper_parameters['steps_per_epoch'] *
                         hyper_parameters['n_exploration_steps'])

        env = normalize(GymEnv(env_id))

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddGaussianNoise(
            env.spec,
            policy,
            total_timesteps=num_timesteps,
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

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        td3 = TD3(env.spec,
                  policy=policy,
                  qf=qf,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  steps_per_epoch=hyper_parameters['steps_per_epoch'],
                  policy_lr=hyper_parameters['policy_lr'],
                  qf_lr=hyper_parameters['qf_lr'],
                  target_update_tau=hyper_parameters['tau'],
                  n_train_steps=hyper_parameters['n_train_steps'],
                  discount=hyper_parameters['discount'],
                  min_buffer_size=hyper_parameters['min_buffer_size'],
                  buffer_batch_size=hyper_parameters['buffer_batch_size'],
                  exploration_policy=exploration_policy,
                  policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                  qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(td3, env)
        trainer.train(n_epochs=hyper_parameters['n_epochs'],
                      batch_size=hyper_parameters['n_exploration_steps'])

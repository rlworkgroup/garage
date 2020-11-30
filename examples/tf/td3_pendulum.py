#!/usr/bin/env python3
"""This is an example to train a task with TD3 algorithm.

Here, we create a gym environment InvertedDoublePendulum
and use a TD3 with 1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import TD3
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer


@wrap_experiment(snapshot_mode='last')
def td3_pendulum(ctxt=None, seed=1):
    """Wrap TD3 training task in the run_task function.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        n_epochs = 500
        steps_per_epoch = 20
        sampler_batch_size = 250
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = GymEnv('InvertedDoublePendulum-v2')

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddGaussianNoise(env.spec,
                                              policy,
                                              total_timesteps=num_timesteps,
                                              max_sigma=0.1,
                                              min_sigma=0.1)

        qf = ContinuousMLPQFunction(name='ContinuousMLPQFunction',
                                    env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    action_merge_layer=0,
                                    hidden_nonlinearity=tf.nn.relu)

        qf2 = ContinuousMLPQFunction(name='ContinuousMLPQFunction2',
                                     env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     action_merge_layer=0,
                                     hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        td3 = TD3(env_spec=env.spec,
                  policy=policy,
                  policy_lr=1e-4,
                  qf_lr=1e-3,
                  qf=qf,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  target_update_tau=1e-2,
                  steps_per_epoch=steps_per_epoch,
                  n_train_steps=1,
                  discount=0.99,
                  buffer_batch_size=100,
                  min_buffer_size=1e4,
                  exploration_policy=exploration_policy,
                  policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                  qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(td3, env)
        trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


td3_pendulum(seed=1)

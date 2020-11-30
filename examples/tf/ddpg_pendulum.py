#!/usr/bin/env python3
"""This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer


@wrap_experiment
def ddpg_pendulum(ctxt=None, seed=1):
    """Train DDPG with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('InvertedDoublePendulum-v2')

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=20,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(algo=ddpg, env=env)

        trainer.train(n_epochs=500, batch_size=100)


ddpg_pendulum(seed=1)

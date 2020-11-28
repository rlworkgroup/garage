#!/usr/bin/env python3
"""This is an example to train a task with DDPG + HER algorithm.

Here it creates a gym environment FetchReach.
"""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import HERReplayBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer


@wrap_experiment(snapshot_mode='last')
def her_ddpg_fetchreach(ctxt=None, seed=1):
    """Train DDPG + HER on the goal-conditioned FetchReach env.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('FetchReach-v1')

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name='Policy',
            hidden_sizes=[256, 256, 256],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            name='QFunction',
            hidden_sizes=[256, 256, 256],
            hidden_nonlinearity=tf.nn.relu,
        )

        # pylint: disable=no-member
        replay_buffer = HERReplayBuffer(capacity_in_transitions=int(1e6),
                                        replay_k=4,
                                        reward_fn=env.compute_reward,
                                        env_spec=env.spec)

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        ddpg = DDPG(
            env_spec=env.spec,
            policy=policy,
            policy_lr=1e-3,
            qf_lr=1e-3,
            qf=qf,
            replay_buffer=replay_buffer,
            sampler=sampler,
            target_update_tau=0.01,
            steps_per_epoch=50,
            n_train_steps=40,
            discount=0.95,
            exploration_policy=exploration_policy,
            policy_optimizer=tf.compat.v1.train.AdamOptimizer,
            qf_optimizer=tf.compat.v1.train.AdamOptimizer,
            buffer_batch_size=256,
        )

        trainer.setup(algo=ddpg, env=env)

        trainer.train(n_epochs=50, batch_size=256)


her_ddpg_fetchreach()

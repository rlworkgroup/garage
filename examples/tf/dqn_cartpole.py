#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.tf.algos import DQN
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction


@wrap_experiment
def dqn_cartpole(ctxt=None, seed=1):
    """Train TRPO with CubeCrash-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        n_epochs = 10
        steps_per_epoch = 10
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
        env = GymEnv('CartPole-v0')
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e4))
        qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(64, 64))
        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.02,
                                                 decay_ratio=0.1)
        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_policy=exploration_policy,
                   max_episode_length=100,
                   replay_buffer=replay_buffer,
                   steps_per_epoch=steps_per_epoch,
                   qf_lr=1e-4,
                   discount=1.0,
                   min_buffer_size=int(1e3),
                   double_q=True,
                   n_train_steps=500,
                   target_network_update_freq=1,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_cartpole()

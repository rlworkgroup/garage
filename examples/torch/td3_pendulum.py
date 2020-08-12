#!/usr/bin/env python3
"""An example to train TD3 algorithm on InvertedDoublePendulum PyTorch."""
import gym
import torch
from torch.nn import functional as F

# from garage.np.exploration_policies import AddGaussianNoise
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@wrap_experiment(snapshot_mode='last')
def td3_pendulum(ctxt=None, seed=1):
    """Train TD3 with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    runner = LocalRunner(ctxt)
    env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        # hidden_sizes=[400, 300],
        hidden_nonlinearity=F.relu,
        output_nonlinearity=torch.tanh)

    # exploration_policy = AddGaussianNoise(env.spec,
    #                                       policy,
    #                                       max_sigma=0.1,
    #                                       min_sigma=0.1)

    qf1 = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        # hidden_sizes=[400, 300],
        hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        # hidden_sizes=[400, 300],
        hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    td3 = TD3(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        #   exploration_policy=exploration_policy,
        policy_optimizer=torch.optim.Adam,
        qf_optimizer=torch.optim.Adam,
        target_update_tau=0.005,
        discount=0.99,
        policy_noise_clip=0.5,
        policy_noise=0.2,
        policy_lr=1e-3,
        qf_lr=1e-3,
        steps_per_epoch=400,  #20
        grad_steps_per_env_step=1,  #1000, #1
        max_episode_length=1000,
        min_buffer_size=int(1e4),
        buffer_batch_size=100)

    runner.setup(algo=td3, env=env)
    runner.train(n_epochs=50, batch_size=100)
    # runner.train(n_epochs=500, batch_size=250)
    # runner.train(n_epochs=1000, batch_size=100)


td3_pendulum()

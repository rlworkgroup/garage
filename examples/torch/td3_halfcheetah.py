#!/usr/bin/env python3
"""An example to train TD3 algorithm on InvertedDoublePendulum PyTorch."""
import torch
from torch.nn import functional as F

# from garage.np.exploration_policies import AddGaussianNoise
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddGaussianNoise
from garage.np.policies import UniformRandomPolicy
from garage.replay_buffer import PathBuffer
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer


@wrap_experiment(snapshot_mode='None')
def td3_half_cheetah(ctxt=None, seed=1):
    """Train TD3 with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
        determinism.
    """
    set_seed(seed)

    n_epochs = 500
    steps_per_epoch = 20
    sampler_batch_size = 250
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

    trainer = Trainer(ctxt)
    env = normalize(GymEnv('HalfCheetah-v2'))

    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    exploration_policy = AddGaussianNoise(env.spec,
                                          policy,
                                          total_timesteps=num_timesteps,
                                          max_sigma=0.1,
                                          min_sigma=0.1)

    uniform_random_policy = UniformRandomPolicy(env.spec)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    td3 = TD3(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              replay_buffer=replay_buffer,
              policy_optimizer=torch.optim.Adam,
              qf_optimizer=torch.optim.Adam,
              exploration_policy=exploration_policy,
              uniform_random_policy=uniform_random_policy,
              target_update_tau=0.005,
              discount=0.99,
              policy_noise_clip=0.5,
              policy_noise=0.2,
              policy_lr=1e-3,
              qf_lr=1e-3,
              steps_per_epoch=40,
              start_steps=1000,
              grad_steps_per_env_step=50,
              min_buffer_size=1000,
              buffer_batch_size=100)

    trainer.setup(algo=td3, env=env)
    trainer.train(n_epochs=750, batch_size=100)


td3_half_cheetah(seed=0)

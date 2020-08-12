#!/usr/bin/env python3
"""This is an example to train a task with DDPG algorithm written in PyTorch.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

"""
import torch
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.torch.algos import DDPG
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@wrap_experiment(snapshot_mode='last')
def ddpg_pendulum(ctxt=None, seed=1, lr=1e-4):
    """Train DDPG with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        lr (float): Learning rate for policy optimization.

    """
    set_seed(seed)
    runner = LocalRunner(ctxt)
    env = normalize(GymEnv('InvertedDoublePendulum-v2'))

    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec, policy, sigma=0.2)

    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    policy_optimizer = (torch.optim.Adagrad, {'lr': lr, 'lr_decay': 0.99})

    ddpg = DDPG(env_spec=env.spec,
                policy=policy,
                qf=qf,
                replay_buffer=replay_buffer,
                max_episode_length=100,
                steps_per_epoch=20,
                n_train_steps=50,
                min_buffer_size=int(1e4),
                exploration_policy=exploration_policy,
                target_update_tau=1e-2,
                discount=0.9,
                policy_optimizer=policy_optimizer,
                qf_optimizer=torch.optim.Adam)

    runner.setup(algo=ddpg, env=env)

    runner.train(n_epochs=500, batch_size=100)


ddpg_pendulum()

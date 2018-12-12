import unittest

import gym
import numpy as np
from torch import nn as nn

from garage.torch.algos.dqn.dqn import DQN
from garage.torch.algos.networks import Mlp
import garage.torch.algos.pytorch_util as ptu


class TestDQN(unittest.TestCase):
    def test_dqn(self):
        variant = dict(
            algo_params=dict(
                num_epochs=1,
                num_steps_per_epoch=10,
                num_steps_per_eval=10,
                batch_size=5,
                max_path_length=10,
                discount=0.99,
                epsilon=0.2,
                tau=0.001,
                hard_update_period=10,
                # Can't serialize CartPole for some reason
                save_environment=False,
            ), )

        env = gym.make('CartPole-v0')
        training_env = gym.make('CartPole-v0')

        qf = Mlp(
            hidden_sizes=[32, 32],
            input_size=int(np.prod(env.observation_space.shape)),
            output_size=env.action_space.n,
        )
        qf_criterion = nn.MSELoss()
        # Use this to switch to DoubleDQN
        # algorithm = DoubleDQN(
        algorithm = DQN(
            env,
            training_env=training_env,
            qf=qf,
            qf_criterion=qf_criterion,
            **variant['algo_params'])
        algorithm.to(ptu.device)
        algorithm.train()

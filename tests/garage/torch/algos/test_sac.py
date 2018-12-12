import unittest

from gym.envs.mujoco import HalfCheetahEnv
import numpy as np

from garage.torch.algos.networks import FlattenMlp
import garage.torch.algos.pytorch_util as ptu
from garage.torch.algos.sac.sac import SoftActorCritic
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.policies import TanhGaussianPolicy


class TestSAC(unittest.TestCase):
    def test_sac(self):
        variant = dict(
            algo_params=dict(
                num_epochs=1,
                num_steps_per_epoch=10,
                num_steps_per_eval=10,
                batch_size=10,
                max_path_length=10,
                discount=0.99,
                reward_scale=1,
                soft_target_tau=0.001,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
            ),
            net_size=300,
        )

        env = NormalizedBoxEnv(HalfCheetahEnv())
        # Or for a specific version:
        # import gym
        # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        net_size = variant['net_size']
        qf = FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        vf = FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim,
            output_size=1,
        )
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        algorithm = SoftActorCritic(
            env=env, policy=policy, qf=qf, vf=vf, **variant['algo_params'])
        algorithm.to(ptu.device)
        algorithm.train()

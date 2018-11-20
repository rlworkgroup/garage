import unittest

from gym.envs.mujoco import HopperEnv

import garage.torch.algos.pytorch_util as ptu
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from garage.torch.exploration_strategies.gaussian_strategy import GaussianStrategy
from garage.torch.launchers.launcher_util import setup_logger
from garage.torch.algos.networks import FlattenMlp, TanhMlpPolicy
from garage.torch.algos.td3.td3 import TD3


class TestTD3(unittest.TestCase):
    def test_td3(self):
        variant = dict(
            algo_kwargs=dict(
                num_epochs=1,
                num_steps_per_epoch=10,
                num_steps_per_eval=10,
                max_path_length=10,
                batch_size=10,
                discount=0.99,
                replay_buffer_size=int(1E6),
            ), )
        setup_logger('common-test', variant=variant)

        env = NormalizedBoxEnv(HopperEnv())
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
        obs_dim = env.observation_space.low.size
        action_dim = env.action_space.low.size
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[400, 300],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[400, 300],
        )
        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=[400, 300],
        )
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        algorithm = TD3(
            env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy,
            **variant['algo_kwargs'])
        algorithm.to(ptu.device)
        algorithm.train()

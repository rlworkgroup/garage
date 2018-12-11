"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
from gym.envs.mujoco import HalfCheetahEnv

from garage.experiment import run_experiment
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy)
from garage.torch.exploration_strategies.ou_strategy import OUStrategy
from garage.torch.algos.networks import FlattenMlp
from garage.torch.policies import TanhMlpPolicy
from garage.torch.algos.ddpg.ddpg import DDPG
import garage.torch.algos.pytorch_util as ptu


def experiment(variant):
    env = NormalizedBoxEnv(HalfCheetahEnv())
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))
    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
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
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params'])
    algorithm.to(ptu.device)
    algorithm.train()


def run_task(*_):
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ), )
    experiment(variant)


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random
    # seed will be used
    seed=1,
    # plot=True,
)

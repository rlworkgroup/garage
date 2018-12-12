"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
from gym.envs.mujoco import HopperEnv

from garage.experiment import run_experiment
from garage.torch.algos.networks import FlattenMlp
import garage.torch.algos.pytorch_util as ptu
from garage.torch.algos.td3.td3 import TD3
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.exploration_strategies.base \
    import PolicyWrappedWithExplorationStrategy
from garage.torch.exploration_strategies.gaussian_strategy \
    import GaussianStrategy
from garage.torch.policies import TanhMlpPolicy


def experiment(variant):
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


def run_task(*_):
    variant = dict(
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=5000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            batch_size=100,
            discount=0.99,
            replay_buffer_size=int(1E6),
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

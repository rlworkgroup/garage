from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.envs.gym_env_util import spec
from rllab.envs.mujoco import SwimmerEnv
from rllab.policies import GaussianMLPPolicy

env = normalize(SwimmerEnv())

policy = GaussianMLPPolicy(
    env_spec=spec(env),
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=spec(env))

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=500,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    # plot=True
)
algo.train()

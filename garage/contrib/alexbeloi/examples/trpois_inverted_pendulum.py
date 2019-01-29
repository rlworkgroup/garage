"""
Example using TRPO with ISSampler, iterations alternate between live and
importance sampled iterations.
"""
import gym

from garage.baselines import LinearFeatureBaseline
from garage.contrib.alexbeloi.is_sampler import ISSampler
from garage.envs import normalize
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy

env = normalize(gym.make('InvertedPendulum-v2'))

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

optimizer_args = dict(
    # debug_nan=True,
    # reg_coeff=0.1,
    # cg_iters=2
)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=200,
    discount=0.99,
    step_size=0.01,
    sampler_cls=ISSampler,
    sampler_args=dict(n_backtrack=1),
    optimizer_args=optimizer_args)
algo.train()

from rllab.algos import TRPO
from rllab.algos import TNPG
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs import normalize
from rllab.policies import GaussianMLPPolicy
from contrib.alexbeloi.is_sampler import ISSampler

"""
Example using VPG with ISSampler, iterations alternate between live and
importance sampled iterations.
"""

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

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
    optimizer_args=optimizer_args
)
algo.train()

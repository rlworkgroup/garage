from contrib.alexbeloi.is_sampler import ISSampler
from garage.algos import VPG
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.policies import GaussianMLPPolicy
"""
Example using VPG with ISSampler, iterations alternate between live and
importance sampled iterations.
"""

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    sampler_cls=ISSampler,
    sampler_args=dict(n_backtrack=1),
)
algo.train()

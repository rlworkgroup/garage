from garage.algos import NOP
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.util import spec
from garage.policies import UniformControlPolicy

env = normalize(CartpoleEnv())

policy = UniformControlPolicy(env_spec=spec(env), )

baseline = ZeroBaseline(env_spec=spec(env))

algo = NOP(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()

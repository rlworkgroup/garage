from garage.algos import NOP
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.theano.envs import TheanoEnv
from garage.theano.policies import UniformControlPolicy

env = TheanoEnv(normalize(CartpoleEnv()))

policy = UniformControlPolicy(env_spec=env.spec, )

baseline = ZeroBaseline(env_spec=env.spec)

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

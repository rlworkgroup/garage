from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.policies import GaussianMLPPolicy
from garage.theano.envs import TheanoEnv

env = TheanoEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    # plot=True
)
algo.train()

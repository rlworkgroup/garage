from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.mujoco import SwimmerEnv
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy

env = TheanoEnv(normalize(SwimmerEnv()))

policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

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

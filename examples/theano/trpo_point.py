from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.point_env import PointEnv
from garage.policies import GaussianMLPPolicy
from garage.theano.envs import TheanoEnv

env = TheanoEnv(normalize(PointEnv()))
policy = GaussianMLPPolicy(env_spec=env.spec, )
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()

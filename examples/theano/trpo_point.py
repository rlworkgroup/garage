from examples.theano.point_env import PointEnv
from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.util import spec
from garage.policies import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(env_spec=spec(env), )
baseline = LinearFeatureBaseline(env_spec=spec(env))
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()

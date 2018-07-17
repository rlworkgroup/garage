import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from nose2 import tools

from garage.algos import VPG
from garage.baselines import GaussianMLPBaseline
from garage.baselines import LinearFeatureBaseline
from garage.baselines import ZeroBaseline
from garage.envs.box2d import CartpoleEnv
from garage.policies import GaussianMLPPolicy
from garage.theano.envs import TheanoEnv

baselines = [ZeroBaseline, LinearFeatureBaseline, GaussianMLPBaseline]


@tools.params(*baselines)
def test_baseline(baseline_cls):
    env = TheanoEnv(CartpoleEnv())
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(6, ))
    baseline = baseline_cls(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        n_itr=1,
        batch_size=1000,
        max_path_length=100)
    algo.train()

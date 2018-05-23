import os

os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from rllab.algos import VPG
from rllab.envs.box2d import CartpoleEnv
from rllab.baselines import ZeroBaseline
from rllab.baselines import LinearFeatureBaseline
from rllab.baselines import GaussianMLPBaseline
from rllab.policies import GaussianMLPPolicy
from nose2 import tools


baselines = [ZeroBaseline, LinearFeatureBaseline, GaussianMLPBaseline]


@tools.params(*baselines)
def test_baseline(baseline_cls):
    env = CartpoleEnv()
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo = VPG(
        env=env, policy=policy, baseline=baseline,
        n_itr=1, batch_size=1000, max_path_length=100
    )
    algo.train()

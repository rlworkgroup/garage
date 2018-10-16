import os
os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'
import unittest

from nose2 import tools
import numpy as np

from garage.algos import CEM
from garage.algos import CMAES
from garage.baselines import ZeroBaseline
from garage.envs import GridWorldEnv
from garage.envs.box2d import CartpoleEnv
from garage.theano.algos import ERWR
from garage.theano.algos import PPO
from garage.theano.algos import REPS
from garage.theano.algos import TNPG
from garage.theano.algos import TRPO
from garage.theano.algos import VPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import CategoricalGRUPolicy
from garage.theano.policies import CategoricalMLPPolicy
from garage.theano.policies import GaussianGRUPolicy
from garage.theano.policies import GaussianMLPPolicy

common_batch_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    VPG:
    common_batch_algo_args,
    TNPG:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(cg_iters=1, ),
    ),
    TRPO:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(cg_iters=1, ),
    ),
    PPO:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(max_penalty_itr=1, max_opt_itr=1),
    ),
    REPS:
    dict(
        common_batch_algo_args,
        max_opt_itr=1,
    ),
    CEM:
    dict(
        n_itr=1,
        max_path_length=100,
        n_samples=5,
    ),
    CMAES:
    dict(
        n_itr=1,
        max_path_length=100,
        batch_size=1000,
    ),
    ERWR:
    common_batch_algo_args,
}

polopt_cases = []
for algo in [VPG, TNPG, PPO, TRPO, CEM, CMAES, ERWR, REPS]:
    polopt_cases.extend([
        (algo, GridWorldEnv, CategoricalMLPPolicy),
        (algo, CartpoleEnv, GaussianMLPPolicy),
        (algo, GridWorldEnv, CategoricalGRUPolicy),
        (algo, CartpoleEnv, GaussianGRUPolicy),
    ])


class TestAlgos(unittest.TestCase):
    @tools.params(*polopt_cases)
    def test_polopt_algo(self, algo_cls, env_cls, policy_cls):
        print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__,
                                      policy_cls.__name__))
        env = TheanoEnv(env_cls())
        policy = policy_cls(env_spec=env.spec)
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = algo_cls(
            env=env,
            policy=policy,
            baseline=baseline,
            **(algo_args.get(algo_cls, dict())))
        algo.train()
        assert not np.any(np.isnan(policy.get_param_values()))

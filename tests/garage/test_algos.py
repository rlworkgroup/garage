import unittest

from nose2 import tools
import numpy as np

from garage.algos import CEM
from garage.algos import CMAES
from garage.baselines import ZeroBaseline
from garage.envs import GridWorldEnv
from garage.envs.box2d import CartpoleEnv
from garage.policies import UniformControlPolicy

common_batch_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    CEM: dict(
        n_itr=1,
        max_path_length=100,
        n_samples=5,
    ),
    CMAES: dict(
        n_itr=1,
        max_path_length=100,
        batch_size=1000,
    )
}

polopt_cases = []
for algo in [CEM, CMAES]:
    polopt_cases.extend([
        (algo, GridWorldEnv, UniformControlPolicy),
        (algo, CartpoleEnv, UniformControlPolicy),
    ])


class TestAlgos(unittest.TestCase):
    @tools.params(*polopt_cases)
    def test_polopt_algo(self, algo_cls, env_cls, policy_cls):
        print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__,
                                      policy_cls.__name__))
        env = env_cls()
        policy = policy_cls(env_spec=env)
        baseline = ZeroBaseline(env_spec=env)
        algo = algo_cls(
            env=env,
            policy=policy,
            baseline=baseline,
            **(algo_args.get(algo_cls, dict())))
        algo.train()
        assert not np.any(np.isnan(policy.get_param_values()))

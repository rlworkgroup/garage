import unittest

from nose2 import tools
import numpy as np

from garage.algos import CEM
from garage.algos import CMAES
from garage.algos.nop import NOP
from garage.baselines import LinearFeatureBaseline
from garage.baselines import ZeroBaseline
from garage.envs import GarageEnv
from garage.envs import GridWorldEnv
from garage.envs import PointEnv
from tests.fixtures.policies import DummyPolicy
from tests.fixtures.policies import DummyRecurrentPolicy

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
        batch_size=5000,
    ),
    NOP: common_batch_algo_args,
}

polopt_cases = []
for algo in [CEM, CMAES, NOP]:
    polopt_cases.extend([
        (algo, GridWorldEnv, DummyPolicy, ZeroBaseline),
        (algo, PointEnv, DummyPolicy, ZeroBaseline),
        (algo, GridWorldEnv, DummyRecurrentPolicy, ZeroBaseline),
        (algo, PointEnv, DummyRecurrentPolicy, ZeroBaseline),
        (algo, GridWorldEnv, DummyPolicy, LinearFeatureBaseline),
        (algo, PointEnv, DummyPolicy, LinearFeatureBaseline),
    ])


class TestAlgos(unittest.TestCase):
    @tools.params(*polopt_cases)
    def test_polopt_algo(self, algo_cls, env_cls, policy_cls, baseline_cls):
        print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__,
                                      policy_cls.__name__))
        env = GarageEnv(env_cls())
        policy = policy_cls(env_spec=env)
        baseline = baseline_cls(env_spec=env)
        algo = algo_cls(
            env=env,
            policy=policy,
            baseline=baseline,
            **(algo_args.get(algo_cls, dict())))
        algo.train()
        assert not np.any(np.isnan(policy.get_param_values()))
        env.close()

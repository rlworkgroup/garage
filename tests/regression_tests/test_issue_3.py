


from nose2.tools import such
from rllab.envs.box2d import CartpoleEnv
from rllab.policies import GaussianMLPPolicy
from rllab.algos import TRPO
from rllab.baselines import ZeroBaseline

with such.A("Issue #3") as it:
    @it.should("be fixed")
    def test_issue_3():
        """
        As reported in https://github.com/rllab/rllab/issues/3, the adaptive_std parameter was not functioning properly
        """
        env = CartpoleEnv()
        policy = GaussianMLPPolicy(
            env_spec=env,
            adaptive_std=True
        )
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=100,
            n_itr=1
        )
        algo.train()

it.createTests(globals())

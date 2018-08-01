from nose2.tools import such

from garage.baselines import ZeroBaseline
from garage.envs.box2d import CartpoleEnv
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy

with such.A("adaptive_std") as it:

    @it.should
    def test_adaptive_std():
        """
        Checks if the adaptive_std parameter works.
        """
        env = TheanoEnv(CartpoleEnv())
        policy = GaussianMLPPolicy(env_spec=env, adaptive_std=True)
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env, policy=policy, baseline=baseline, batch_size=100, n_itr=1)
        algo.train()


# This call generates the unittest.TestCase instances for all of the tests
it.createTests(globals())

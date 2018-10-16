import os
os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.theano.algos import DDPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import DeterministicMLPPolicy
from garage.theano.q_functions import ContinuousMLPQFunction


class TestDDPG(unittest.TestCase):
    def test_ddpg(self):
        env = TheanoEnv(CartpoleEnv())
        policy = DeterministicMLPPolicy(env.spec)
        qf = ContinuousMLPQFunction(env.spec)
        es = OUStrategy(env.spec)
        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1000), time_horizon=100)
        algo = DDPG(
            env=env,
            policy=policy,
            qf=qf,
            es=es,
            pool=replay_buffer,
            n_epochs=1,
            epoch_length=100,
            batch_size=32,
            min_pool_size=50,
            eval_samples=100,
        )
        algo.train()

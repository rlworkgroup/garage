from garage.envs import GymEnv
from garage.tf.algos.rl2 import RL2Worker

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.policies import DummyPolicy


class TestRL2Worker(TfGraphTestCase):

    def test_rl2_worker(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(1, )))
        policy = DummyPolicy(env_spec=env.spec)
        worker = RL2Worker(seed=1,
                           max_episode_length=100,
                           worker_number=1,
                           n_episodes_per_trial=5)
        worker.update_agent(policy)
        worker.update_env(env)
        episodes = worker.rollout()
        assert episodes.rewards.shape[0] == 500

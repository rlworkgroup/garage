from garage.envs import GarageEnv
from garage.tf.algos.rl2 import RL2Worker
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.policies import DummyPolicy


class TestRL2Worker(TfGraphTestCase):

    def test_rl2_worker(self):
        env = GarageEnv(DummyBoxEnv(obs_dim=(1, )))
        policy = DummyPolicy(env_spec=env.spec)
        worker = RL2Worker(seed=1,
                           max_path_length=100,
                           worker_number=1,
                           n_paths_per_trial=5)
        worker.update_agent(policy)
        worker.update_env(env)
        rollouts = worker.rollout()
        assert rollouts.rewards.shape[0] == 500

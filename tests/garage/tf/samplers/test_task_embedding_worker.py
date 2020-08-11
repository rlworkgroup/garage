from unittest.mock import Mock

import numpy as np

from garage.envs import GymEnv
from garage.tf.algos.te import TaskEmbeddingWorker

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestTaskEmbeddingWorker(TfGraphTestCase):

    def test_task_embedding_worker(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(1, )))
        env.active_task_one_hot = np.array([1., 0., 0., 0.])
        env._active_task_one_hot = lambda: np.array([1., 0., 0., 0.])

        a = np.random.random(env.action_space.shape)
        z = np.random.random(5)
        latent_info = dict(mean=np.random.random(5))
        agent_info = dict(dummy='dummy')

        policy = Mock()
        policy.get_latent.return_value = (z, latent_info)
        policy.latent_space.flatten.return_value = z
        policy.get_action_given_latent.return_value = (a, agent_info)

        worker = TaskEmbeddingWorker(seed=1,
                                     max_episode_length=100,
                                     worker_number=1)
        worker.update_agent(policy)
        worker.update_env(env)

        episodes = worker.rollout()
        assert 'task_onehot' in episodes.env_infos
        assert np.array_equal(episodes.env_infos['task_onehot'][0],
                              env.active_task_one_hot)
        assert 'latent' in episodes.agent_infos
        assert np.array_equal(episodes.agent_infos['latent'][0], z)
        assert 'latent_mean' in episodes.agent_infos
        assert np.array_equal(episodes.agent_infos['latent_mean'][0],
                              latent_info['mean'])

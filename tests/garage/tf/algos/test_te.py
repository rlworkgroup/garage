"""This script tests Task Embedding algorithms."""
import akro
import numpy as np
import tensorflow as tf

from garage import InOutSpec
from garage.envs import MultiEnvWrapper, PointEnv
from garage.envs.multi_env_wrapper import round_robin_strategy
from garage.experiment import LocalTFRunner
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TEPPO
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTE(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()

        def circle(r, n):
            """Generate n points on a circle of radius r.

            Args:
                r (float): Radius of the circle.
                n (int): Number of points to generate.

            Yields:
                tuple(float, float): Coordinate of a point.

            """
            for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
                yield r * np.sin(t), r * np.cos(t)

        N = 4
        goals = circle(3.0, N)
        tasks = {
            str(i + 1): {
                'args': [],
                'kwargs': {
                    'goal': g,
                    'never_done': False,
                    'done_bonus': 0.0,
                }
            }
            for i, g in enumerate(goals)
        }

        latent_length = 1
        inference_window = 2
        self.batch_size = 100 * len(tasks)
        self.policy_ent_coeff = 2e-2
        self.encoder_ent_coeff = 2.2e-3
        self.inference_ce_coeff = 5e-2
        self.max_episode_length = 100
        embedding_init_std = 1.0
        embedding_max_std = 2.0
        embedding_min_std = 0.38
        policy_init_std = 1.0
        policy_max_std = None
        policy_min_std = None

        task_names = sorted(tasks.keys())
        task_args = [tasks[t]['args'] for t in task_names]
        task_kwargs = [tasks[t]['kwargs'] for t in task_names]

        task_envs = [
            PointEnv(*t_args, **t_kwargs)
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        self.env = env = MultiEnvWrapper(task_envs,
                                         round_robin_strategy,
                                         mode='vanilla')

        latent_lb = np.zeros(latent_length, )
        latent_ub = np.ones(latent_length, )
        latent_space = akro.Box(latent_lb, latent_ub)

        obs_lb, obs_ub = env.observation_space.bounds
        obs_lb_flat = env.observation_space.flatten(obs_lb)
        obs_ub_flat = env.observation_space.flatten(obs_ub)
        traj_lb = np.stack([obs_lb_flat] * inference_window)
        traj_ub = np.stack([obs_ub_flat] * inference_window)
        traj_space = akro.Box(traj_lb, traj_ub)

        task_embed_spec = InOutSpec(env.task_space, latent_space)
        traj_embed_spec = InOutSpec(traj_space, latent_space)

        self.inference = GaussianMLPEncoder(
            name='inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=[20, 10],
            std_share_network=True,
            init_std=2.0,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        task_encoder = GaussianMLPEncoder(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=[20, 20],
            std_share_network=True,
            init_std=embedding_init_std,
            max_std=embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        self.policy = GaussianMLPTaskEmbeddingPolicy(
            name='policy',
            env_spec=env.spec,
            encoder=task_encoder,
            hidden_sizes=[32, 16],
            std_share_network=True,
            max_std=policy_max_std,
            init_std=policy_init_std,
            min_std=policy_min_std,
        )

        self.baseline = LinearMultiFeatureBaseline(
            env_spec=env.spec, features=['observations', 'tasks', 'latents'])

    def test_te_worker(self):
        worker = TaskEmbeddingWorker(seed=1,
                                     max_episode_length=100,
                                     worker_number=1)
        worker.update_env(self.env)
        worker.update_agent(self.policy)
        worker.start_episode()
        while not worker.step_episode():
            pass
        episodes = worker.collect_episode()
        assert 'task_onehot' in episodes.env_infos.keys()
        assert episodes.env_infos['task_onehot'][0].shape == (4, )
        assert 'latent' in episodes.agent_infos.keys()
        assert episodes.agent_infos['latent'][0].shape == (1, )

    def test_te_ppo(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TEPPO(env_spec=self.env.spec,
                         policy=self.policy,
                         baseline=self.baseline,
                         inference=self.inference,
                         max_episode_length=self.max_episode_length,
                         discount=0.99,
                         lr_clip_range=0.2,
                         policy_ent_coeff=self.policy_ent_coeff,
                         encoder_ent_coeff=self.encoder_ent_coeff,
                         inference_ce_coeff=self.inference_ce_coeff,
                         use_softplus_entropy=True,
                         optimizer_args=dict(
                             batch_size=32,
                             max_episode_length=10,
                         ),
                         inference_optimizer_args=dict(
                             batch_size=32,
                             max_episode_length=10,
                         ),
                         center_adv=True,
                         stop_ce_gradient=True)

            runner.setup(algo,
                         self.env,
                         sampler_cls=LocalSampler,
                         sampler_args=None,
                         worker_class=TaskEmbeddingWorker)
            runner.train(n_epochs=1, batch_size=self.batch_size, plot=False)

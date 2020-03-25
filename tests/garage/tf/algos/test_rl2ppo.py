"""
This script creates a test that fails when garage.tf.algos.RL2PPO
performance is too low.
"""
import pytest
try:
    # pylint: disable=unused-import
    import mujoco_py  # noqa: F401
except ImportError:
    pytest.skip('To use mujoco-based features, please install garage[mujoco].',
                allow_module_level=True)
except Exception:  # pylint: disable=broad-except
    pytest.skip(
        'Skipping tests, failed to import mujoco. Do you have a '
        'valid mujoco key installed?',
        allow_module_level=True)

from garage.envs import normalize
from garage.envs.mujoco import HalfCheetahDirEnv
from garage.experiment import task_sampler
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestRL2PPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.max_path_length = 100
        self.meta_batch_size = 10
        self.episode_per_task = 4
        self.tasks = task_sampler.SetTaskSampler(lambda: RL2Env(env=normalize(
            HalfCheetahDirEnv())))
        self.env_spec = RL2Env(env=normalize(HalfCheetahDirEnv())).spec
        self.policy = GaussianGRUPolicy(env_spec=self.env_spec,
                                        hidden_dim=64,
                                        state_include_action=False)
        self.baseline = LinearFeatureBaseline(env_spec=self.env_spec)

    def test_ppo_pendulum(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = RL2PPO(rl2_max_path_length=self.max_path_length,
                          meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
                          pg_loss='surrogate_clip',
                          optimizer_args=dict(
                              batch_size=32,
                              max_epochs=10,
                          ),
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          max_path_length=self.max_path_length *
                          self.episode_per_task)

            runner.setup(algo,
                         self.tasks.sample(self.meta_batch_size),
                         sampler_cls=LocalSampler,
                         n_workers=self.meta_batch_size,
                         worker_class=RL2Worker)

            last_avg_ret = runner.train(n_epochs=10,
                                        batch_size=self.episode_per_task *
                                        self.max_path_length *
                                        self.meta_batch_size)
            assert last_avg_ret > -40

    def test_ppo_pendulum_wrong_worker(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            with pytest.raises(ValueError):
                algo = RL2PPO(rl2_max_path_length=self.max_path_length,
                              meta_batch_size=self.meta_batch_size,
                              task_sampler=self.tasks,
                              env_spec=self.env_spec,
                              policy=self.policy,
                              baseline=self.baseline,
                              discount=0.99,
                              gae_lambda=0.95,
                              lr_clip_range=0.2,
                              pg_loss='surrogate_clip',
                              optimizer_args=dict(
                                  batch_size=32,
                                  max_epochs=10,
                              ),
                              stop_entropy_gradient=True,
                              entropy_method='max',
                              policy_ent_coeff=0.02,
                              center_adv=False,
                              max_path_length=self.max_path_length *
                              self.episode_per_task,
                              flatten_input=False)

                runner.setup(algo,
                             self.tasks.sample(self.meta_batch_size),
                             sampler_cls=LocalSampler,
                             n_workers=self.meta_batch_size)

                runner.train(n_epochs=10,
                             batch_size=self.episode_per_task *
                             self.max_path_length * self.meta_batch_size)

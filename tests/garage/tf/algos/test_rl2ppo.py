"""
This script creates a test that fails when garage.tf.algos.RL2PPO
performance is too low.
"""
import numpy as np
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
from garage.experiment import LocalTFRunner
from garage.experiment import task_sampler
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


@pytest.mark.mujoco
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

    @pytest.mark.timeout(120)
    def test_rl2_ppo_pendulum(self):
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
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          max_path_length=self.max_path_length *
                          self.episode_per_task)

            runner.setup(
                algo,
                self.tasks.sample(self.meta_batch_size),
                sampler_cls=LocalSampler,
                n_workers=self.meta_batch_size,
                worker_class=RL2Worker,
                worker_args=dict(n_paths_per_trial=self.episode_per_task))

            last_avg_ret = runner.train(n_epochs=1,
                                        batch_size=self.episode_per_task *
                                        self.max_path_length *
                                        self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_ppo_pendulum_meta_test(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            meta_evaluator = MetaEvaluator(
                test_task_sampler=self.tasks,
                n_exploration_traj=10,
                n_test_rollouts=10,
                max_path_length=self.max_path_length,
                n_test_tasks=1)

            algo = RL2PPO(rl2_max_path_length=self.max_path_length,
                          meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
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
                          meta_evaluator=meta_evaluator,
                          n_epochs_per_eval=10)

            runner.setup(algo,
                         self.tasks.sample(self.meta_batch_size),
                         sampler_cls=LocalSampler,
                         n_workers=self.meta_batch_size,
                         worker_class=RL2Worker)

            last_avg_ret = runner.train(n_epochs=1,
                                        batch_size=self.episode_per_task *
                                        self.max_path_length *
                                        self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_ppo_pendulum_exploration_policy(self):
        with LocalTFRunner(snapshot_config, sess=self.sess):
            algo = RL2PPO(rl2_max_path_length=self.max_path_length,
                          meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
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
            exploration_policy = algo.get_exploration_policy()
            params = exploration_policy.get_param_values()
            new_params = np.zeros_like(params)
            exploration_policy.set_param_values(new_params)
            assert np.array_equal(new_params,
                                  exploration_policy.get_param_values())

    def test_rl2_ppo_pendulum_adapted_policy(self):
        with LocalTFRunner(snapshot_config, sess=self.sess):
            algo = RL2PPO(rl2_max_path_length=self.max_path_length,
                          meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
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
            exploration_policy = algo.get_exploration_policy()
            adapted_policy = algo.adapt_policy(exploration_policy, [])
            (params, hidden) = adapted_policy.get_param_values()
            expected_new_params = np.zeros_like(params)
            expected_hidden = np.zeros_like(hidden)
            adapted_policy.set_param_values(
                (expected_new_params, expected_hidden))
            (new_params, new_hidden) = adapted_policy.get_param_values()
            assert np.array_equal(expected_new_params, new_params)
            assert np.array_equal(expected_hidden, new_hidden)

    def test_rl2_ppo_pendulum_wrong_worker(self):
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

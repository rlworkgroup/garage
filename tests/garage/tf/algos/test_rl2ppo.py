"""
This script creates a test that fails when garage.tf.algos.RL2PPO
performance is too low.
"""
# yapf: disable
import numpy as np
import pytest

from garage.envs import GymEnv, normalize
from garage.experiment import task_sampler
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable

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

from garage.envs.mujoco import HalfCheetahDirEnv  # isort:skip


@pytest.mark.mujoco
class TestRL2PPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.meta_batch_size = 10
        self.episode_per_task = 4
        self.max_episode_length = 100
        # Avoid pickling self
        max_episode_length = 100
        self.tasks = task_sampler.SetTaskSampler(
            HalfCheetahDirEnv,
            wrapper=lambda env, _: RL2Env(
                normalize(GymEnv(env, max_episode_length=max_episode_length))))
        self.env_spec = RL2Env(
            normalize(
                GymEnv(HalfCheetahDirEnv(),
                       max_episode_length=max_episode_length))).spec
        self.policy = GaussianGRUPolicy(env_spec=self.env_spec,
                                        hidden_dim=64,
                                        state_include_action=False)
        self.baseline = LinearFeatureBaseline(env_spec=self.env_spec)
        self.sampler = LocalSampler(
            agents=self.policy,
            envs=self.tasks.sample(self.meta_batch_size),
            max_episode_length=self.env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=self.meta_batch_size,
            worker_class=RL2Worker)

    def test_rl2_ppo_pendulum(self):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            sampler = LocalSampler(
                agents=self.policy,
                envs=self.tasks.sample(self.meta_batch_size),
                max_episode_length=self.env_spec.max_episode_length,
                is_tf_worker=True,
                n_workers=self.meta_batch_size,
                worker_class=RL2Worker,
                worker_args=dict(n_episodes_per_trial=self.episode_per_task))
            algo = RL2PPO(meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          sampler=sampler,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          episodes_per_trial=self.episode_per_task)

            trainer.setup(algo, self.tasks.sample(self.meta_batch_size))

            last_avg_ret = trainer.train(n_epochs=1,
                                         batch_size=self.episode_per_task *
                                         self.max_episode_length *
                                         self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_ppo_pendulum_meta_test(self):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            meta_evaluator = MetaEvaluator(test_task_sampler=self.tasks,
                                           n_exploration_eps=10,
                                           n_test_episodes=10,
                                           n_test_tasks=1)

            algo = RL2PPO(meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          sampler=self.sampler,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
                          optimizer_args=dict(
                              batch_size=32,
                              max_optimization_epochs=10,
                          ),
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          episodes_per_trial=self.episode_per_task,
                          meta_evaluator=meta_evaluator,
                          n_epochs_per_eval=10)

            trainer.setup(algo, self.tasks.sample(self.meta_batch_size))

            last_avg_ret = trainer.train(n_epochs=1,
                                         batch_size=self.episode_per_task *
                                         self.max_episode_length *
                                         self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_ppo_pendulum_exploration_policy(self):
        with TFTrainer(snapshot_config, sess=self.sess):
            algo = RL2PPO(meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          sampler=self.sampler,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
                          optimizer_args=dict(
                              batch_size=32,
                              max_optimization_epochs=10,
                          ),
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          episodes_per_trial=self.episode_per_task)

            exploration_policy = algo.get_exploration_policy()
            params = exploration_policy.get_param_values()
            new_params = np.zeros_like(params)
            exploration_policy.set_param_values(new_params)
            assert np.array_equal(new_params,
                                  exploration_policy.get_param_values())

    def test_rl2_ppo_pendulum_adapted_policy(self):
        with TFTrainer(snapshot_config, sess=self.sess):
            algo = RL2PPO(meta_batch_size=self.meta_batch_size,
                          task_sampler=self.tasks,
                          env_spec=self.env_spec,
                          policy=self.policy,
                          baseline=self.baseline,
                          sampler=self.sampler,
                          discount=0.99,
                          gae_lambda=0.95,
                          lr_clip_range=0.2,
                          optimizer_args=dict(
                              batch_size=32,
                              max_optimization_epochs=10,
                          ),
                          stop_entropy_gradient=True,
                          entropy_method='max',
                          policy_ent_coeff=0.02,
                          center_adv=False,
                          episodes_per_trial=self.episode_per_task)

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
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            with pytest.raises(ValueError):
                sampler = LocalSampler(
                    agents=self.policy,
                    envs=self.tasks.sample(self.meta_batch_size),
                    max_episode_length=self.env_spec.max_episode_length,
                    is_tf_worker=True,
                    n_workers=self.meta_batch_size)
                algo = RL2PPO(meta_batch_size=self.meta_batch_size,
                              task_sampler=self.tasks,
                              env_spec=self.env_spec,
                              policy=self.policy,
                              baseline=self.baseline,
                              sampler=sampler,
                              discount=0.99,
                              gae_lambda=0.95,
                              lr_clip_range=0.2,
                              optimizer_args=dict(
                                  batch_size=32,
                                  max_optimization_epochs=10,
                              ),
                              stop_entropy_gradient=True,
                              entropy_method='max',
                              policy_ent_coeff=0.02,
                              center_adv=False,
                              episodes_per_trial=self.episode_per_task)

                trainer.setup(algo, self.tasks.sample(self.meta_batch_size))

                trainer.train(n_epochs=10,
                              batch_size=self.episode_per_task *
                              self.max_episode_length * self.meta_batch_size)

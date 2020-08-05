"""
This script creates a test that fails when garage.tf.algos.RL2TRPO
performance is too low.
"""
# yapf: disable
import pytest

from garage.envs import GymEnv, normalize
from garage.experiment import LocalTFRunner, task_sampler
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2TRPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHvp,
                                  PenaltyLbfgsOptimizer)
from garage.tf.policies import GaussianGRUPolicy

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
class TestRL2TRPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.max_episode_length = 100
        self.meta_batch_size = 10
        self.episode_per_task = 4
        self.tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            normalize(GymEnv(HalfCheetahDirEnv()))))
        self.env_spec = RL2Env(normalize(GymEnv(HalfCheetahDirEnv()))).spec
        self.policy = GaussianGRUPolicy(env_spec=self.env_spec,
                                        hidden_dim=64,
                                        state_include_action=False)
        self.baseline = LinearFeatureBaseline(env_spec=self.env_spec)

    def test_rl2_trpo_pendulum(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = RL2TRPO(
                rl2_max_episode_length=self.max_episode_length,
                meta_batch_size=self.meta_batch_size,
                task_sampler=self.tasks,
                env_spec=self.env_spec,
                policy=self.policy,
                baseline=self.baseline,
                max_episode_length=self.max_episode_length *
                self.episode_per_task,
                discount=0.99,
                max_kl_step=0.01,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                    base_eps=1e-5)))

            runner.setup(algo,
                         self.tasks.sample(self.meta_batch_size),
                         sampler_cls=LocalSampler,
                         n_workers=self.meta_batch_size,
                         worker_class=RL2Worker)

            last_avg_ret = runner.train(n_epochs=1,
                                        batch_size=self.episode_per_task *
                                        self.max_episode_length *
                                        self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_trpo_pendulum_default_optimizer(self):
        with LocalTFRunner(snapshot_config, sess=self.sess):
            algo = RL2TRPO(rl2_max_episode_length=self.max_episode_length,
                           meta_batch_size=self.meta_batch_size,
                           task_sampler=self.tasks,
                           env_spec=self.env_spec,
                           policy=self.policy,
                           baseline=self.baseline,
                           kl_constraint='hard',
                           max_episode_length=self.max_episode_length *
                           self.episode_per_task,
                           discount=0.99,
                           max_kl_step=0.01)
            assert isinstance(algo._inner_algo._optimizer,
                              ConjugateGradientOptimizer)

    def test_ppo_pendulum_default_optimizer2(self):
        with LocalTFRunner(snapshot_config, sess=self.sess):
            algo = RL2TRPO(rl2_max_episode_length=self.max_episode_length,
                           meta_batch_size=self.meta_batch_size,
                           task_sampler=self.tasks,
                           env_spec=self.env_spec,
                           policy=self.policy,
                           baseline=self.baseline,
                           kl_constraint='soft',
                           max_episode_length=self.max_episode_length *
                           self.episode_per_task,
                           discount=0.99,
                           max_kl_step=0.01)
            assert isinstance(algo._inner_algo._optimizer,
                              PenaltyLbfgsOptimizer)

    def test_rl2_trpo_pendulum_invalid_kl_constraint(self):
        with LocalTFRunner(snapshot_config, sess=self.sess):
            with pytest.raises(ValueError):
                RL2TRPO(rl2_max_episode_length=self.max_episode_length,
                        meta_batch_size=self.meta_batch_size,
                        task_sampler=self.tasks,
                        env_spec=self.env_spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        kl_constraint='xyz',
                        max_episode_length=self.max_episode_length *
                        self.episode_per_task,
                        discount=0.99,
                        max_kl_step=0.01)

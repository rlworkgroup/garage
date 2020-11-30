"""
This script creates a test that fails when garage.tf.algos.RL2TRPO
performance is too low.
"""
# yapf: disable
import pytest

from garage.envs import GymEnv, normalize
from garage.experiment import task_sampler
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2TRPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHVP, PenaltyLBFGSOptimizer)
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
class TestRL2TRPO(TfGraphTestCase):

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

    def test_rl2_trpo_pendulum(self):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            algo = RL2TRPO(
                meta_batch_size=self.meta_batch_size,
                task_sampler=self.tasks,
                env_spec=self.env_spec,
                policy=self.policy,
                baseline=self.baseline,
                sampler=self.sampler,
                episodes_per_trial=self.episode_per_task,
                discount=0.99,
                max_kl_step=0.01,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                    base_eps=1e-5)))

            trainer.setup(algo, self.tasks.sample(self.meta_batch_size))

            last_avg_ret = trainer.train(n_epochs=1,
                                         batch_size=self.episode_per_task *
                                         self.max_episode_length *
                                         self.meta_batch_size)
            assert last_avg_ret > -40

    def test_rl2_trpo_pendulum_default_optimizer(self):
        with TFTrainer(snapshot_config, sess=self.sess):
            algo = RL2TRPO(meta_batch_size=self.meta_batch_size,
                           task_sampler=self.tasks,
                           env_spec=self.env_spec,
                           policy=self.policy,
                           baseline=self.baseline,
                           sampler=self.sampler,
                           kl_constraint='hard',
                           episodes_per_trial=self.episode_per_task,
                           discount=0.99,
                           max_kl_step=0.01)
            assert isinstance(algo._inner_algo._optimizer,
                              ConjugateGradientOptimizer)

    def test_ppo_pendulum_default_optimizer2(self):
        with TFTrainer(snapshot_config, sess=self.sess):
            algo = RL2TRPO(meta_batch_size=self.meta_batch_size,
                           task_sampler=self.tasks,
                           env_spec=self.env_spec,
                           policy=self.policy,
                           baseline=self.baseline,
                           sampler=self.sampler,
                           kl_constraint='soft',
                           episodes_per_trial=self.episode_per_task,
                           discount=0.99,
                           max_kl_step=0.01)
            assert isinstance(algo._inner_algo._optimizer,
                              PenaltyLBFGSOptimizer)

    def test_rl2_trpo_pendulum_invalid_kl_constraint(self):
        with TFTrainer(snapshot_config, sess=self.sess):
            with pytest.raises(ValueError):
                RL2TRPO(meta_batch_size=self.meta_batch_size,
                        task_sampler=self.tasks,
                        env_spec=self.env_spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        sampler=self.sampler,
                        kl_constraint='xyz',
                        episodes_per_trial=self.episode_per_task,
                        discount=0.99,
                        max_kl_step=0.01)

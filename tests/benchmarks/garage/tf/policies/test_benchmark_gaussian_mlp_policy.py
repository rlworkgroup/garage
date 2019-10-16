import gym
import numpy as np
import pytest
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import deterministic, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy


class TestBenchmarkGaussianMLPPolicy:
    '''Compare benchmarks between garage and baselines.'''

    def setup_method(self):
        self._env = None
        self._seed = None

    @pytest.mark.huge
    def test_benchmark_gaussian_mlp_policy(self):
        bench_envs = [
            'HalfCheetah-v2', 'Reacher-v2', 'Walker2d-v2', 'Hopper-v2',
            'Swimmer-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2'
        ]

        seeds = np.random.choice(100, size=(len(bench_envs), 3))

        for env_num in range(len(bench_envs)):
            self._env = bench_envs[env_num]
            for seed in seeds[env_num]:
                self._seed = seed
                deterministic.set_seed(self._seed)
                name = '{}_seed_{}_garage'.format(self._env, self._seed)
                run_experiment(self.run_task,
                               snapshot_mode='last',
                               seed=self._seed,
                               n_parallel=12,
                               exp_name=name)

    def run_task(self, snapshot_config, *_):
        config = tf.ConfigProto(device_count={'GPU': 0},
                                allow_soft_placement=True,
                                intra_op_parallelism_threads=12,
                                inter_op_parallelism_threads=12)
        sess = tf.Session(config=config)
        with LocalTFRunner(snapshot_config=snapshot_config,
                           sess=sess) as runner:
            env = gym.make(self._env)
            env = TfEnv(normalize(env))
            env.reset()
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )

            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(
                    hidden_sizes=(64, 64),
                    use_trust_region=False,
                    optimizer=FirstOrderOptimizer,
                    optimizer_args=dict(
                        batch_size=32,
                        max_epochs=10,
                        tf_optimizer_args=dict(learning_rate=1e-3),
                    ),
                ),
            )

            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                policy_ent_coeff=0.0,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,
                    tf_optimizer_args=dict(learning_rate=1e-3),
                ),
            )
            runner.setup(algo, env, sampler_args=dict(n_envs=12))
            runner.train(n_epochs=5, batch_size=2048)

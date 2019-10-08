"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import pytest

from garage.envs import normalize
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestPPOWithModel(TfGraphTestCase):

    @pytest.mark.large
    def test_ppo_pendulum_gru_with_model(self):
        """Test PPO with Pendulum environment and GRU policy."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            policy = GaussianGRUPolicy(env_spec=env.spec, )
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
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
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80

            env.close()

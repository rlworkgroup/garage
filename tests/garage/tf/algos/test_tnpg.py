import pytest

from garage.envs import GymEnv, normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TNPG
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTNPG(TfGraphTestCase):

    @pytest.mark.mujoco_long
    def test_tnpg_inverted_pendulum(self):
        """Test TNPG with InvertedPendulum-v2 environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(GymEnv('InvertedPendulum-v2'))

            policy = GaussianMLPPolicy(name='policy',
                                       env_spec=env.spec,
                                       hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TNPG(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=0.99,
                        optimizer_args=dict(reg_coeff=5e-1))

            trainer.setup(algo, env)

            last_avg_ret = trainer.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 15

            env.close()

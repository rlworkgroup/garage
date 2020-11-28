import pytest

from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import VPG
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestVPG(TfGraphTestCase):

    @pytest.mark.large
    def test_vpg_cartpole(self):
        """Test VPG with CartPole-v1 environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       sampler=sampler,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env)

            last_avg_ret = trainer.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 90

            env.close()

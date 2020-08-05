import pytest

from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import VPG
from garage.tf.policies import CategoricalMLPPolicy

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestVPG(TfGraphTestCase):

    @pytest.mark.large
    def test_vpg_cartpole(self):
        """Test VPG with CartPole-v1 environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_episode_length=100,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            runner.setup(algo, env, sampler_cls=LocalSampler)

            last_avg_ret = runner.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 90

            env.close()

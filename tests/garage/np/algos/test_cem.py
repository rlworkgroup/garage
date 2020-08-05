import pytest

from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.np.algos import CEM
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestCEM(TfGraphTestCase):

    @pytest.mark.large
    def test_cem_cartpole(self):
        """Test CEM with Cartpole-v1 environment."""
        with LocalTFRunner(snapshot_config) as runner:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))
            baseline = LinearFeatureBaseline(env_spec=env.spec)

            n_samples = 10

            algo = CEM(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       best_frac=0.1,
                       max_episode_length=100,
                       n_samples=n_samples)

            runner.setup(algo, env, sampler_cls=LocalSampler)
            rtn = runner.train(n_epochs=10, batch_size=2048)
            assert rtn > 40

            env.close()

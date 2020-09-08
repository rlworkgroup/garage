import pytest

from garage.envs import GymEnv
from garage.np.algos import CEM
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestCEM(TfGraphTestCase):

    @pytest.mark.large
    def test_cem_cartpole(self):
        """Test CEM with Cartpole-v1 environment."""
        with TFTrainer(snapshot_config) as trainer:
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
                       n_samples=n_samples)

            trainer.setup(algo, env, sampler_cls=LocalSampler)
            rtn = trainer.train(n_epochs=10, batch_size=2048)
            assert rtn > 40

            env.close()

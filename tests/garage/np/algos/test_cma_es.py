from garage.envs import GymEnv
from garage.np.algos import CMAES
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestCMAES(TfGraphTestCase):

    def test_cma_es_cartpole(self):
        """Test CMAES with Cartpole-v1 environment."""
        with TFTrainer(snapshot_config) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))
            baseline = LinearFeatureBaseline(env_spec=env.spec)

            n_samples = 20

            algo = CMAES(env_spec=env.spec,
                         policy=policy,
                         baseline=baseline,
                         n_samples=n_samples)

            trainer.setup(algo, env, sampler_cls=LocalSampler)
            trainer.train(n_epochs=1, batch_size=1000)
            # No assertion on return because CMAES is not stable.

            env.close()

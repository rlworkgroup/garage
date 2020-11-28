from garage.envs import GymEnv
from garage.np.algos import CMAES
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

            n_samples = 20

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = CMAES(env_spec=env.spec,
                         policy=policy,
                         sampler=sampler,
                         n_samples=n_samples)

            trainer.setup(algo, env)
            trainer.train(n_epochs=1, batch_size=1000)
            # No assertion on return because CMAES is not stable.

            env.close()

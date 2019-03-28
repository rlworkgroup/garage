from garage.algos import CMAES
from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import OnPolicyVectorizedSampler
from tests.fixtures import TfGraphTestCase


class TestCMAES(TfGraphTestCase):
    def test_cma_es_cartpole(self):
        """Test CMAES with Cartpole-v1 environment."""
        with LocalRunner() as runner:
            env = TfEnv(env_name="CartPole-v1")

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            runner.initialize_tf_vars()

            n_samples = 20

            algo = CMAES(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                n_samples=n_samples)

            runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
            runner.train(n_epochs=1, batch_size=1000, n_epoch_cycles=n_samples)
            # No assertion on return because CMAES is not stable.

            env.close()

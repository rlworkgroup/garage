from garage.experiment import LocalRunner
from garage.np.algos import CEM
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import OnPolicyVectorizedSampler
from tests.fixtures import TfGraphTestCase


class TestCEM(TfGraphTestCase):
    def test_cem_cartpole(self):
        """Test CEM with Cartpole-v1 environment."""
        with LocalRunner() as runner:
            env = TfEnv(env_name="CartPole-v1")

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            runner.initialize_tf_vars()

            n_samples = 10

            algo = CEM(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                best_frac=0.1,
                max_path_length=100,
                n_samples=n_samples)

            runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
            rtn = runner.train(
                n_epochs=5, batch_size=2000, n_epoch_cycles=n_samples)
            assert rtn > 40

            env.close()

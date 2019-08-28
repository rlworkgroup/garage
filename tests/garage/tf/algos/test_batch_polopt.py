import gym

from garage.envs import normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.envs import TfEnv
from garage.tf.samplers import BatchSampler
from tests.fixtures.algos import DummyTFAlgo
from tests.fixtures.policies import DummyPolicy, DummyPolicyWithoutVectorized


class TestBatchPolopt:

    def setup_method(self):
        self.env = TfEnv(normalize(gym.make('CartPole-v1')))
        self.baseline = LinearFeatureBaseline(env_spec=self.env.spec)

    def test_default_sampler_cls(self):
        policy = DummyPolicy(env_spec=self.env.spec)
        algo = DummyTFAlgo(env_spec=self.env.spec,
                           policy=policy,
                           baseline=self.baseline)
        sampler = algo.sampler_cls(algo, self.env, dict())
        assert isinstance(sampler, OnPolicyVectorizedSampler)

        policy = DummyPolicyWithoutVectorized(env_spec=self.env.spec)
        algo = DummyTFAlgo(env_spec=self.env.spec,
                           policy=policy,
                           baseline=self.baseline)
        sampler = algo.sampler_cls(algo, self.env, dict())
        assert isinstance(sampler, BatchSampler)

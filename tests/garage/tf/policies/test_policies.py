import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalGRUPolicy
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


class TestDiscretePolicies(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = TfEnv(DummyDiscreteEnv())

    def teardown_method(self):
        self.env.close()
        super().teardown_method()

    def test_categorial_gru_policy(self):
        categorical_gru_policy = CategoricalGRUPolicy(env_spec=self.env,
                                                      hidden_dim=1)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        categorical_gru_policy.reset()

        obs = self.env.observation_space.high
        assert categorical_gru_policy.get_action(obs)

    def test_categorical_lstm_policy(self):
        categorical_lstm_policy = CategoricalLSTMPolicy(env_spec=self.env,
                                                        hidden_dim=1)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        categorical_lstm_policy.reset()

        obs = self.env.observation_space.high
        assert categorical_lstm_policy.get_action(obs)

    def test_categorial_mlp_policy(self):
        categorical_mlp_policy = CategoricalMLPPolicy(env_spec=self.env,
                                                      hidden_sizes=(1, ))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        obs = self.env.observation_space.high
        assert categorical_mlp_policy.get_action(obs)


class TestContinuousPolicies(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = TfEnv(DummyBoxEnv())

    def teardown_method(self):
        self.env.close()
        super().teardown_method()

    def test_continuous_mlp_policy(self):
        continuous_mlp_policy = ContinuousMLPPolicy(env_spec=self.env,
                                                    hidden_sizes=(1, ))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        obs = self.env.observation_space.high
        assert continuous_mlp_policy.get_action(obs)

    def test_gaussian_gru_policy(self):
        gaussian_gru_policy = GaussianGRUPolicy(env_spec=self.env,
                                                hidden_dim=1)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        gaussian_gru_policy.reset()

        obs = self.env.observation_space.high
        assert gaussian_gru_policy.get_action(obs)

    def test_gaussian_lstm_policy(self):
        gaussian_lstm_policy = GaussianLSTMPolicy(env_spec=self.env,
                                                  hidden_dim=1)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        gaussian_lstm_policy.reset()

        obs = self.env.observation_space.high
        assert gaussian_lstm_policy.get_action(obs)

    def test_gaussian_mlp_policy(self):
        gaussian_mlp_policy = GaussianMLPPolicy(env_spec=self.env,
                                                hidden_sizes=(1, ))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        obs = self.env.observation_space.high
        assert gaussian_mlp_policy.get_action(obs)

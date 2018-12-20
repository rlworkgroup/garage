import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.distributions import DiagonalGaussian
from garage.tf.models import GaussianMLPModel
from garage.tf.policies import StochasticPolicy
from garage.tf.spaces import Box


class GaussianMLPPolicy(StochasticPolicy, Parameterized, Serializable):

    def __init__(self,
                 env_spec,
                 name="GaussianMLPPolicy",
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 mean_network=None,
                 std_network=None,
                 std_parameterization='exp'):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden
        layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers
         for std
        :param min_std: whether to make sure that the std is at least some
         threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parameterization: how the std should be parametrized. There
         are a few options:
            - exp: the logarithm of the std will be stored, and applied a
             exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        assert isinstance(env_spec.action_space, Box)
        Serializable.quick_init(self, locals())
        super(GaussianMLPPolicy, self).__init__(env_spec)
        Parameterized.__init__(self)
        self.name = name
        
        self.model = GaussianMLPModel(
            input_dim=env_spec.observation_space.flat_dim,
            output_dim=env_spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            learn_std=learn_std,
            init_std=init_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            std_hidden_sizes=std_hidden_sizes,
            min_std=min_std,
            max_std=max_std,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            std_parameterization=std_parameterization,
        )

        self._dist = DiagonalGaussian(self.env_spec.action_space.flat_dim)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        flat_obs = self.observation_space.flatten(observation)
        feed_dict = {self.model.inputs: flat_obs}
        action, mean, std = sess.run(
            [self.model.sample, self.model.mean, self.model.std], 
            feed_dict=feed_dict
        )
        return action, dict(mean=mean, log_std=std) 

    def get_actions(self, observations, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        flat_obs = self.observation_space.flatten_n(observations)
        feed_dict = {self.model.inputs: flat_obs}
        actions, means, stds = sess.run(
            [self.model.sample, self.model.mean, self.model.std_param], 
            feed_dict=feed_dict
        )
        return actions, dict(mean=means, log_std=stds)

    @property
    def distribution(self):
        return self._dist
    
    def log_diagnostics(self, paths):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular("{}/AverageStd".format(self.name),
                              np.mean(np.exp(log_stds)))

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        with tf.name_scope(name, "dist_info_sym", [obs_var, state_info_vars]):
            _, outputs, _ = self.model.build_model(inputs=obs_var)
            mean_var = outputs["mean"]
            log_std = outputs["std_param"]
        return dict(mean=mean_var, log_std=log_std)

    @overrides
    def get_params_internal(self, **tags):
        return self.model.get_params_internal(**tags)

    def entropy_sym(self, obs_var, name=None):
        with tf.name_scope(name, "entropy_sym", [obs_var]):
            _, _, info = self.model.build_model(inputs=obs_var)
            dist = info["dist"]
        return dist.entropy()

    @property
    def inputs(self):
        return self.model.inputs

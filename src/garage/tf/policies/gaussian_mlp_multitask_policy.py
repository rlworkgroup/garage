"""GaussianMLPMultitaskPolicy."""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from garage.tf.policies.multitask_policy import StochasticMultitaskPolicy


class GaussianMLPMultitaskPolicy(StochasticMultitaskPolicy):
    """GaussianMLPMultitaskPolicy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        embedding (garage.tf.embeddings.Embedding): Embedding network.
        task_space (akro.Box): Space of the task.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        learn_std (bool): Is std trainable.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_output_nonlinearity (callable): Nonlinearity for output layer in
            the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_parameterization (str): How the std should be parametrized. There
            are a few options:
            - exp: the logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.

    """
    def __init__(self,
                 env_spec,
                 embedding,
                 task_space,
                 name='GaussianMLPMultitaskPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        assert isinstance(env_spec.action_space, akro.Box)
        super().__init__(env_spec, embedding, task_space, name)
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

        self.model = GaussianMLPModel(
            output_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            name='GaussianMLPModel')

        self._initialize()

    def _initialize(self):
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, self.obs_dim))
        task_input = self._embedding.input
        latent_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._embedding.latent_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs

            with tf.variable_scope('concat_latent_obs'):
                latent_state_input = tf.concat(
                    [latent_input, state_input], axis=-1)
            self.model.build(latent_state_input, name='from_latent')

            # Connect with embedding network's latent output
            with tf.variable_scope('concat_embed_obs'):
                latent_dist_info_sym = self._embedding.dist_info_sym(
                    task_input, name='dist_info_sym')
                latent_var = self._embedding.distribution.sample_sym(
                    latent_dist_info_sym)

                embed_state_input = tf.concat(
                    [latent_var, state_input], axis=-1)
            self.model.build(embed_state_input, name='default')

        self._f_dist_latent_obs = tf.compat.v1.get_default_session().make_callable(
            [
                self.model.networks['from_latent'].mean,
                self.model.networks['from_latent'].log_std
            ],
            feed_list=[latent_input, state_input])

        self._f_dist_task_obs = tf.compat.v1.get_default_session().make_callable(
            [
                self.model.networks['default'].mean,
                self.model.networks['default'].log_std,
                self._embedding.latent_mean,
                self._embedding.latent_std_param,
            ],
            feed_list=[task_input, state_input])

    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """
        flat_task_obs = self.task_observation_space.flatten(observation)
        flat_task, flat_obs = self.split_observation(flat_task_obs)

        (action_mean, action_log_std, latent_mean, latent_log_std) = self._f_dist_task_obs([flat_task], [flat_obs])

        rnd = np.random.normal(size=action_mean.shape)
        action_sample = rnd * np.exp(action_log_std) + action_mean
        action_sample = self.action_space.unflatten(action_sample[0])
        action_mean = self.action_space.unflatten(action_mean[0])
        action_log_std = self.action_space.unflatten(action_log_std[0])

        mean = self._embedding.latent_space.unflatten(latent_mean[0])
        log_std = self._embedding.latent_space.unflatten(latent_log_std[0])
        latent_info = dict(mean=latent_mean, log_std=latent_log_std)
        return action, dict(mean=action_mean, log_std=action_log_std, latent_info=latent_info)

    def get_action_from_latent(self, latent, observation):
        """Get action sampled from the latent and observation.

        Args:
            latent (np.ndarray): Latent var from the policy.
            observation (np.ndarray): Observation from the environment.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """
        flat_obs = self.observation_space.flatten(observation)
        flat_latent = self.latent_space.flatten(latent)

        mean, log_std = self._f_dist_latent_obs([flat_latent], [flat_obs])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def dist_info_sym(self, task_var, obs_var, state_info_vars=None, name='default'):
        """Build a symbolic graph of the distribution parameters.

        Args:
            task_var (tf.Tensor): Tensor input for symbolic graph.
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            latent_dist_info_sym = self._embedding.dist_info_sym(
                task_var, name=name)
            latent = self._embedding.distribution.sample_sym(
                latent_dist_info_sym)
            embed_state_input = tf.concat(
                [latent, obs_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(embed_state_input, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def dist_info_sym_from_latent(self, latent_var, obs_var, state_info_vars=None,
                                  name='from_latent'):
        """Build a symbolic graph of the distribution parameters from latent.

        Args:
            latent_var (tf.Tensor): Tensor input for symbolic graph.
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            embed_state_input = tf.concat([latent_var, obs_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(embed_state_input, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    @property
    def distribution(self):
        """Policy distribution.

        Returns:
            garage.tf.distributions.DiagonalGaussian: Policy distribution.

        """
        return self.model.networks['default'].dist

    def get_action_from_onehot(self, observation, onehot):
        """Get action sampled from the policy based on onehot index.

        Args:
            observation (np.ndarray): Observation from the environment.
            onehot (np.ndarray): One hot task index.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """
        raise NotImplementedError

    def get_actions_from_onehots(self, observations, onehots):
        """Get actions sampled from the policy based on onehot indices.

        Args:
            observations (np.ndarray): Observations from the environment.
            onehots (np.ndarray): One hot task indices.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """
        raise NotImplementedError

    def get_actions_from_latents(self, observations, latents):
        """Get actions sampled from the policy.

        Args:
            observations (np.ndarray): Observations from the environment.
            latents (np.ndarray): Latent.

        Returns:
            (np.ndarray): Actions sampled from the policy.

        """
        raise NotImplementedError

    def get_actions(self, observations):
        """Get action sampled from the policy.

        Args:
            observations (list[np.ndarray]): Observations from the environment.

        Returns:
            (np.ndarray): Actions sampled from the policy.

        """
        raise NotImplementedError

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_dist_latent_obs']
        del new_dict['_f_dist_task_obs']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()


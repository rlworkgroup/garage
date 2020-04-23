"""GaussianMLPTaskEmbeddingPolicy."""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy


class GaussianMLPTaskEmbeddingPolicy(TaskEmbeddingPolicy):
    """GaussianMLPTaskEmbeddingPolicy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        encoder (garage.tf.embeddings.StochasticEncoder): Embedding network.
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
                 encoder,
                 name='GaussianMLPTaskEmbeddingPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
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
        super().__init__(name, env_spec, encoder)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self.model = GaussianMLPModel(
            output_dim=self._action_dim,
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
        obs_input = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, self._obs_dim))
        task_input = self._encoder.input
        latent_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._encoder.output_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs

            with tf.compat.v1.variable_scope('concat_obs_latent'):
                obs_latent_input = tf.concat([obs_input, latent_input],
                                             axis=-1)
            self.model.build(obs_latent_input, name='given_latent')

            with tf.compat.v1.variable_scope('concat_obs_task'):
                latent_dist_info_sym = self._encoder.dist_info_sym(
                    task_input, name='dist_info_sym')
                latent_var = self._encoder.distribution.sample_sym(
                    latent_dist_info_sym)

                embed_state_input = tf.concat([obs_input, latent_var], axis=-1)
            self.model.build(embed_state_input, name='given_task')

        self._f_dist_obs_latent = tf.compat.v1.get_default_session(
        ).make_callable([
            self.model.networks['given_latent'].mean,
            self.model.networks['given_latent'].log_std
        ],
                        feed_list=[obs_input, latent_input])

        self._f_dist_obs_task = tf.compat.v1.get_default_session(
        ).make_callable([
            self.model.networks['given_task'].mean,
            self.model.networks['given_task'].log_std
        ],
                        feed_list=[obs_input, task_input])

    def dist_info(self, input_val, state_infos=None):
        """Action distribution info.

        Return the distribution information about the actions.

        Args:
            input_val (np.ndarray): Observation values,
                with shape :math:`(O+N, )`. O is the dimension of observation,
                N is the number of tasks.
            state_infos (dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation, with keys
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`. Z is the dimension
                    of action.

        Returns:
            dict[numpy.ndarray]: Action distribution parameters, with keys
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. Z is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`. Z is the dimension
                    of action.

        """
        obs, task = self.split_augmented_observation(input_val)
        mean, log_std = self._f_dist_obs_task([obs], [task])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return dict(mean=mean, log_std=log_std)

    def dist_info_sym(self, input_var, state_info_vars=None, name='default'):
        """Symbolic graph of action distribution.

        Return the symbolic distribution information about the actions.

        Args:
            input_var (tf.Tensor): symbolic variable for augmented
                observations, with shape :math:`(T, O+N)`. T is the number of
                environment steps, O is the dimension of action, N is the
                number of tasks.
            state_info_vars (dict): Extra state information, e.g.
                previous action. It contains the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(T, A)`.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of action
                distribution parameters. It contains the following values,
                - mean (tf.Tensor): Symbolic mean of the distribution, with
                    shape :math:`(T, A)`. T is the number of environment steps,
                    A is the dimension of action.
                - log_std (tf.Tensor): Symbolic log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.

        """
        task_dim = self.task_space.flat_dim
        obs, task = input_var[:, :-task_dim], input_var[:, -task_dim:]
        return self.dist_info_sym_given_task(obs, task, state_info_vars, name)

    def dist_info_sym_given_task(self,
                                 obs_var,
                                 task_var,
                                 state_info_vars=None,
                                 name='given_task'):
        """Build a symbolic graph of the action distribution given task.

        Args:
            obs_var (tf.Tensor): Symbolic observation input,
                with shape :math:`(T, O)`. T is the number of environment
                steps, O is the dimension of observation.
            task_var (tf.Tensor): Symbolic task input,
                with shape :math:`(T, N)`. T is the number of environment
                steps, N is the number of tasks.
            state_info_vars (dict): Extra state information, e.g.
                previous action. It contains the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(T, A)`. T is the number of environment steps,
                    A is the dimension of action.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of action
                distribution parameters. It contains the following values,
                - mean (tf.Tensor): Symbolic mean of the distribution, with
                    shape :math:`(T, A)`. T is the number of environment steps,
                    A is the dimension of action.
                - log_std (tf.Tensor): Symbolic log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            latent_dist_info_sym = self._encoder.dist_info_sym(task_var,
                                                               name=name)
            latent_var = self._encoder.distribution.sample_sym(
                latent_dist_info_sym)
            obs_latent_input = tf.concat([obs_var, latent_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(obs_latent_input,
                                                           name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def dist_info_sym_given_latent(self,
                                   obs_var,
                                   latent_var,
                                   state_info_vars=None,
                                   name='given_latent'):
        """Build a symbolic graph of the action distribution given latent.

        Args:
            obs_var (tf.Tensor): Symbolic observation input,
                with shape :math:`(T, O)`. T is the number of environment
                steps, O is the dimension of observation.
            latent_var (tf.Tensor): Symbolic latent input,
                with shape :math:`(T, Z)`. T is the number of environment
                steps, Z is the dimension of latent embedding.
            state_info_vars (dict): Extra state information, e.g.
                previous action. It contains the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(T, A)`. T is the number of environment steps,
                    A is the dimension of action.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of action
                distribution parameters. It contains the following values,
                - mean (tf.Tensor): Symbolic mean of the distribution, with
                    shape :math:`(T, A)`. T is the number of environment steps,
                    A is the dimension of action.
                - log_std (tf.Tensor): Symbolic log standard deviation of the
                    distribution, with shape :math:`(T, A)`.
                    T is the number of environment steps,
                    A is the dimension of action.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            obs_latent_input = tf.concat([obs_var, latent_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(obs_latent_input,
                                                           name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    @property
    def distribution(self):
        """Policy action distribution.

        Returns:
            garage.tf.distributions.DiagonalGaussian: Policy distribution.

        """
        return self.model.networks['given_latent'].dist

    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Augmented observation from the
                environment, with shape :math:`(O+N, )`. O is the dimension of
                observation, N is the number of tasks.

        Returns:
            np.ndarray: Action sampled from the policy,
                with shape :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`.
                    A is the dimension of action.

        """
        obs, task = self.split_augmented_observation(observation)
        return self.get_action_given_task(obs, task)

    def get_actions(self, observations):
        """Get actions sampled from the policy.

        Args:
            observations (np.ndarray): Augmented observation from the
                environment, with shape :math:`(T, O+N)`. T is the number of
                environment steps, O is the dimension of observation, N is the
                number of tasks.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(T, A)`. T is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(T, A)`. T is the number of environment
                    steps, A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`. T is the number of
                    environment steps, Z is the dimension of action.

        """
        obses, tasks = zip(*[
            self.split_augmented_observation(aug_obs)
            for aug_obs in observations
        ])
        return self.get_actions_given_tasks(np.array(obses), np.array(tasks))

    def get_action_given_latent(self, observation, latent):
        """Sample an action given observation and latent.

        Args:
            observation (np.ndarray): Observation from the environment,
                with shape :math:`(O, )`. O is the dimension of observation.
            latent (np.ndarray): Latent, with shape :math:`(Z, )`. Z is the
                dimension of the latent embedding.

        Returns:
            np.ndarray: Action sampled from the policy,
                with shape :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`. A is the dimension
                    of action.

        """
        flat_obs = self.observation_space.flatten(observation)
        flat_latent = self.latent_space.flatten(latent)

        mean, log_std = self._f_dist_obs_latent([flat_obs], [flat_latent])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def get_actions_given_latents(self, observations, latents):
        """Sample a batch of actions given observations and latents.

        Args:
            observations (np.ndarray): Observations from the environment, with
                shape :math:`(T, O)`. T is the number of environment steps, O
                is the dimension of observation.
            latents (np.ndarray): Latents, with shape :math:`(T, Z)`. T is the
                number of environment steps, Z is the dimension of
                latent embedding.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(T, A)`. T is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, , with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(T, A)`. T is the number of
                    environment steps. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`. T is the number of
                    environment steps. A is the dimension of action.

        """
        flat_obses = self.observation_space.flatten_n(observations)
        flat_latents = self.latent_space.flatten_n(latents)

        means, log_stds = self._f_dist_obs_latent(flat_obses, flat_latents)
        rnds = np.random.normal(size=means.shape)
        samples = rnds * np.exp(log_stds) + means
        samples = self.action_space.unflatten_n(samples)
        means = self.action_space.unflatten_n(means)
        log_stds = self.action_space.unflatten_n(log_stds)
        return samples, dict(mean=means, log_std=log_stds)

    def get_action_given_task(self, observation, task_id):
        """Sample an action given observation and task id.

        Args:
            observation (np.ndarray): Observation from the environment, with
                shape :math:`(O, )`. O is the dimension of the observation.
            task_id (np.ndarray): One-hot task id, with shape :math:`(N, ).
                N is the number of tasks.

        Returns:
            np.ndarray: Action sampled from the policy, with shape
                :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`. A is the dimension
                    of action.

        """
        flat_obs = self.observation_space.flatten(observation)

        mean, log_std = self._f_dist_obs_task([flat_obs], [task_id])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def get_actions_given_tasks(self, observations, task_ids):
        """Sample a batch of actions given observations and task ids.

        Args:
            observations (np.ndarray): Observations from the environment, with
                shape :math:`(T, O)`. T is the number of environment steps,
                O is the dimension of observation.
            task_ids (np.ndarry): One-hot task ids, with shape :math:`(T, N)`.
                T is the number of environment steps, N is the number of tasks.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(T, A)`. T is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, , with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(T, A)`. T is the number of
                    environment steps. A is the dimension of action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(T, A)`. T is the number of
                    environment steps. A is the dimension of action.

        """
        flat_obses = self.observation_space.flatten_n(observations)

        means, log_stds = self._f_dist_obs_task(flat_obses, task_ids)
        rnds = np.random.normal(size=means.shape)
        samples = rnds * np.exp(log_stds) + means
        samples = self.action_space.unflatten_n(samples)
        means = self.action_space.unflatten_n(means)
        log_stds = self.action_space.unflatten_n(log_stds)
        return samples, dict(mean=means, log_std=log_stds)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_dist_obs_latent']
        del new_dict['_f_dist_obs_task']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()

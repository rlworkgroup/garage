"""GaussianMLPTaskEmbeddingPolicy."""
# pylint: disable=wrong-import-order
import akro
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import GaussianMLPModel
from garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy


# pylint: disable=too-many-ancestors
class GaussianMLPTaskEmbeddingPolicy(GaussianMLPModel, TaskEmbeddingPolicy):
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
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
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
        assert not isinstance(env_spec.observation_space, akro.Dict)
        self._env_spec = env_spec
        self._name = name
        self._encoder = encoder
        self._augmented_observation_space = akro.concat(
            self._env_spec.observation_space, self.task_space)
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._learn_std = learn_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._init_std = init_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_sizes = std_hidden_sizes
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_parameterization = std_parameterization
        self._layer_normalization = layer_normalization

        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

        super().__init__(output_dim=self.action_dim,
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
                         name=name)

        self._initialize()

    def _initialize(self):
        """Build policy to support sampling.

        After build, get_action_*() methods will be available.

        """
        obs_input = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, None, self.obs_dim))
        encoder_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, None, self._encoder.input_dim))
        latent_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, None, self._encoder.output_dim))

        with tf.compat.v1.variable_scope(self._encoder.name):
            encoder_dist = self._encoder.build(encoder_input,
                                               name='encoder').dist

        with tf.compat.v1.variable_scope('concat_obs_latent'):
            obs_latent_input = tf.concat([obs_input, latent_input], -1)

        dist, mean_var, log_std_var = super().build(
            obs_latent_input,
            # Must named 'default' to
            # compensate tf default worker
            name='default').outputs

        embed_state_input = tf.concat([
            obs_input,
            encoder_dist.sample(seed=deterministic.get_tf_seed_stream())
        ], -1)
        dist_given_task, mean_g_t, log_std_g_t = super().build(
            embed_state_input, name='given_task').outputs

        self._f_dist_obs_latent = tf.compat.v1.get_default_session(
        ).make_callable([
            dist.sample(seed=deterministic.get_tf_seed_stream()), mean_var,
            log_std_var
        ],
                        feed_list=[obs_input, latent_input])

        self._f_dist_obs_task = tf.compat.v1.get_default_session(
        ).make_callable([
            dist_given_task.sample(seed=deterministic.get_tf_seed_stream()),
            mean_g_t, log_std_g_t
        ],
                        feed_list=[obs_input, encoder_input])

    # pylint: disable=arguments-differ
    def build(self, obs_input, task_input, name=None):
        """Build policy.

        Args:
            obs_input (tf.Tensor): Observation input.
            task_input (tf.Tensor): One-hot task id input.
            name (str): Name of the model, which is also the name scope.

        Returns:
            namedtuple: Policy network.
            namedtuple: Encoder network.

        """
        name = name or 'additional'
        # Encoder should be outside policy scope
        with tf.compat.v1.variable_scope(self._encoder.name):
            enc_net = self._encoder.build(task_input, name=name)
            latent_var = enc_net.dist.loc

        embed_state_input = tf.concat([obs_input, latent_var], -1)
        return super().build(embed_state_input, name=name), enc_net

    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Augmented observation from the
                environment, with shape :math:`(O+N, )`. O is the dimension
                of observation, N is the number of tasks.

        Returns:
            np.ndarray: Action sampled from the policy,
                with shape :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(A, )`. A is the dimension of
                    action.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(A, )`.
                    A is the dimension of action.

        """
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

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
        flat_obs = np.expand_dims([flat_obs], 1)
        flat_latent = self.latent_space.flatten(latent)
        flat_latent = np.expand_dims([flat_latent], 1)

        sample, mean, log_std = self._f_dist_obs_latent(flat_obs, flat_latent)
        sample = self.action_space.unflatten(np.squeeze(sample, 1)[0])
        mean = self.action_space.unflatten(np.squeeze(mean, 1)[0])
        log_std = self.action_space.unflatten(np.squeeze(log_std, 1)[0])
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
        flat_obses = np.expand_dims(flat_obses, 1)
        flat_latents = self.latent_space.flatten_n(latents)
        flat_latents = np.expand_dims(flat_latents, 1)

        samples, means, log_stds = self._f_dist_obs_latent(
            flat_obses, flat_latents)
        samples = self.action_space.unflatten_n(np.squeeze(samples, 1))
        means = self.action_space.unflatten_n(np.squeeze(means, 1))
        log_stds = self.action_space.unflatten_n(np.squeeze(log_stds, 1))
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
        flat_obs = np.expand_dims([flat_obs], 1)
        task_id = np.expand_dims([task_id], 1)

        sample, mean, log_std = self._f_dist_obs_task(flat_obs, task_id)
        sample = self.action_space.unflatten(np.squeeze(sample, 1)[0])
        mean = self.action_space.unflatten(np.squeeze(mean, 1)[0])
        log_std = self.action_space.unflatten(np.squeeze(log_std, 1)[0])
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
        flat_obses = np.expand_dims(flat_obses, 1)
        task_ids = np.expand_dims(task_ids, 1)

        samples, means, log_stds = self._f_dist_obs_task(flat_obses, task_ids)
        samples = self.action_space.unflatten_n(np.squeeze(samples, 1))
        means = self.action_space.unflatten_n(np.squeeze(means, 1))
        log_stds = self.action_space.unflatten_n(np.squeeze(log_stds, 1))
        return samples, dict(mean=means, log_std=log_stds)

    def get_trainable_vars(self):
        """Get trainable variables.

        The trainable vars of a multitask policy should be the trainable vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return (self._variable_scope.trainable_variables() +
                self.encoder.get_trainable_vars())

    def get_global_vars(self):
        """Get global variables.

        The global vars of a multitask policy should be the global vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return (self._variable_scope.global_variables() +
                self.encoder.get_global_vars())

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    @property
    def encoder(self):
        """garage.tf.embeddings.encoder.Encoder: Encoder."""
        return self._encoder

    @property
    def augmented_observation_space(self):
        """akro.Box: Concatenated observation space and one-hot task id."""
        return self._augmented_observation_space

    def clone(self, name):
        """Return a clone of the policy.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.GaussianMLPTaskEmbeddingPolicy: Cloned policy.

        """
        new_policy = self.__class__(
            env_spec=self.env_spec,
            encoder=self.encoder.clone('{}_encoder'.format(name)),
            name=name,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            learn_std=self._learn_std,
            adaptive_std=self._adaptive_std,
            std_share_network=self._std_share_network,
            init_std=self._init_std,
            min_std=self._min_std,
            max_std=self._max_std,
            std_hidden_sizes=self._std_hidden_sizes,
            std_hidden_nonlinearity=self._std_hidden_nonlinearity,
            std_output_nonlinearity=self._std_output_nonlinearity,
            std_parameterization=self._std_parameterization,
            layer_normalization=self._layer_normalization)
        new_policy.parameters = self.parameters
        return new_policy

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

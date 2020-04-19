"""Policy class for Task Embedding envs."""
import abc

import akro

from garage.tf.policies import StochasticPolicy


class TaskEmbeddingPolicy(StochasticPolicy):
    """Base class for Task Embedding policies in TensorFlow.

    This policy needs a task id in addition to observation to sample an action.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.EnvSpec): Environment specification.
        encoder (garage.tf.embeddings.StochasticEncoder):
            A encoder that embeds a task id to a latent.

    """

    # pylint: disable=too-many-public-methods

    def __init__(self, name, env_spec, encoder):
        super().__init__(name, env_spec)
        self._encoder = encoder
        self._augmented_observation_space = akro.concat(
            self._env_spec.observation_space, self.task_space)

    @property
    def encoder(self):
        """garage.tf.embeddings.encoder.Encoder: Encoder."""
        return self._encoder

    def get_latent(self, task_id):
        """Get embedded task id in latent space.

        Args:
            task_id (np.ndarray): One-hot task id, with shape :math:`(N, )`. N
                is the number of tasks.

        Returns:
            np.ndarray: An embedding sampled from embedding distribution, with
                shape :math:`(M, )`. M is the dimension of the latent
                embedding.
            dict: Embedding distribution information, with keys
                - mean (numpy.ndarray): Mean of the distribution, with shape
                    :math:`(M, )`.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`.
                M is the shape of the embedding.

        """
        return self.encoder.forward(task_id)

    @property
    def latent_space(self):
        """akro.Box: Space of latent."""
        return self.encoder.spec.output_space

    @property
    def task_space(self):
        """akro.Box: One-hot space of task id."""
        return self.encoder.spec.input_space

    @property
    def augmented_observation_space(self):
        """akro.Box: Concatenated observation space and one-hot task id."""
        return self._augmented_observation_space

    @property
    def encoder_distribution(self):
        """garage.tf.distributions.DiagonalGaussian: Encoder distribution."""
        return self.encoder.distribution

    @abc.abstractmethod
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
                    with shape :math:`(M, )`. M is the dimension of
                    the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`. M is the dimension
                    of the latent embedding.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions sampled from the policy.

        Args:
            observations (np.ndarray): Augmented observation from the
                environment, with shape :math:`(B, O+N)`. B is the number of
                environment steps, O is the dimension of observation, N is the
                number of tasks.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(B, A)`. B is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(B, M)`. B is the number of environment
                    steps, M is the dimension of the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(B, M)`. B is the number of
                    environment steps, M is the dimension of the latent
                    embedding.

        """

    @abc.abstractmethod
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
                    with shape :math:`(M, )`. M is the dimension of
                    the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`. M is the dimension
                    of the latent embedding.

        """

    @abc.abstractmethod
    def get_actions_given_tasks(self, observations, task_ids):
        """Sample a batch of actions given observations and task ids.

        Args:
            observations (np.ndarray): Observations from the environment, with
                shape :math:`(B, O)`. B is the number of environment steps,
                O is the dimension of observation.
            task_ids (np.ndarry): One-hot task ids, with shape :math:`(B, N)`.
                B is the number of environment steps, N is the number of tasks.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(B, A)`. B is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, , with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(B, M)`. B is the number of
                    environment steps. M is the dimension of
                    the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(B, M)`. B is the number of
                    environment steps. M is the dimension of
                    the latent embedding.

        """

    @abc.abstractmethod
    def get_action_given_latent(self, observation, latent):
        """Sample an action given observation and latent.

        Args:
            observation (np.ndarray): Observation from the environment,
                with shape :math:`(O, )`. O is the dimension of observation.
            latent (np.ndarray): Latent, with shape :math:`(M, )`. M is the
                dimension of latent embedding.

        Returns:
            np.ndarray: Action sampled from the policy,
                with shape :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information, with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(M, )`. M is the dimension of
                    the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`. M is the dimension
                    of the latent embedding.

        """

    @abc.abstractmethod
    def get_actions_given_latents(self, observations, latents):
        """Sample a batch of actions given observations and latents.

        Args:
            observations (np.ndarray): Observations from the environment, with
                shape :math:`(B, O)`. B is the number of environment steps, O
                is the dimension of observation.
            latents (np.ndarray): Latents, with shape :math:`(B, M)`. B is the
                number of environment steps, M is the dimension of
                latent embedding.

        Returns:
            np.ndarray: Actions sampled from the policy,
                with shape :math:`(B, A)`. B is the number of environment
                steps, A is the dimension of action.
            dict: Action distribution information, , with keys:
                - mean (numpy.ndarray): Mean of the distribution,
                    with shape :math:`(B, M)`. B is the number of
                    environment steps. M is the dimension of
                    the latent embedding.
                - log_std (numpy.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(B, M)`. B is the number of
                    environment steps. M is the dimension of
                    the latent embedding.

        """

    def encoder_dist_info_sym(self,
                              input_var,
                              state_info_vars=None,
                              name='encoder_dist_info_sym'):
        """Return the symbolic distribution information about the encoder.

        Args:
            input_var(tf.Tensor): Symbolic variable for encoder input,
                with shape :math:`(None, N)`. N is the number of tasks.
            state_info_vars(dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                receives the input. It contains the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(M, )`.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`.
                M is the shape of the embedding.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of encoder
                distribution parameters. It contains the following values,
                - mean (tf.Tensor): Symbolic mean of the distribution, with
                    shape :math:`(M, )`.
                - log_std (tf.Tensor): Symbolic log standard deviation of the
                    distribution, with shape :math:`(M, )`.
                M is the shape of the embedding.

        """
        return self.encoder.dist_info_sym(input_var, state_info_vars, name)

    def encoder_dist_info(self, input_val, state_infos=None):
        """Return the distribution information about the encoder.

        Args:
            input_val(tf.Tensor): Encoder input values.
            state_infos(dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                receives the input. It contains the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(M, )`.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`.
                M is the shape of the embedding.

        Returns:
            dict[numpy.ndarray]: Encoder distribution parameters. It contains
                the following values,
                - mean (np.ndarray): Mean of the distribution, with shape
                    :math:`(M, )`.
                - log_std (np.ndarray): Log standard deviation of the
                    distribution, with shape :math:`(M, )`.
                M is the shape of the embedding.

        """
        return self.encoder.dist_info(input_val, state_infos)

    @abc.abstractmethod
    def dist_info_sym_given_task(self,
                                 obs_var,
                                 task_var,
                                 state_info_vars=None,
                                 name='default'):
        """Build a symbolic graph of the action distribution given task.

        Args:
            obs_var (tf.Tensor): Symbolic observation input,
                with shape :math:`(None, O)`. O is the dimension
                of observation.
            task_var (tf.Tensor): Symbolic task input,
                with shape :math:`(None, N)`. N is the number of tasks.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of action
                distribution parameters.

        """

    @abc.abstractmethod
    def dist_info_sym_given_latent(self,
                                   obs_var,
                                   latent_var,
                                   state_info_vars=None,
                                   name='given_latent'):
        """Build a symbolic graph of the action distribution given latent.

        Args:
            obs_var (tf.Tensor): Symbolic observation input,
                with shape :math:`(None, O)`. O is the dimension of
                observation.
            latent_var (tf.Tensor): Symbolic latent input,
                with shape :math:`(None, M)`. M is the dimension of
                latent embedding.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """

    def dist_info_sym(self, input_var, state_info_vars=None, name='default'):
        """Symbolic graph of action distribution.

        Return the symbolic distribution information about the actions.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            input_var (tf.Tensor): symbolic variable for observations,
                with shape :math:`(None, O)`. O is the dimension of action.
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        raise NotImplementedError

    def dist_info(self, input_val, state_infos):
        """Action distribution info.

        Return the distribution information about the actions.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            input_val (np.ndarray): Observation values,
                with shape :math:`(O, )`. O is the dimension of ovservation.
            state_infos (dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.

        Returns:
            dict[numpy.ndarray]: Action distribution parameters.

        """
        raise NotImplementedError

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

    def split_augmented_observation(self, collated):
        """Splits up observation into one-hot task and environment observation.

        Args:
            collated (np.ndarray): Environment observation concatenated with
                task one-hot, with shape :math:`(O+N, )`. O is the dimension of
                observation, N is the number of tasks.

        Returns:
            np.ndarray: Vanilla environment observation,
                with shape :math:`(O, )`. O is the dimension of observation.
            np.ndarray: Task one-hot, with shape :math:`(N, )`. N is the number
                of tasks.

        """
        task_dim = self.task_space.flat_dim
        return collated[:-task_dim], collated[-task_dim:]

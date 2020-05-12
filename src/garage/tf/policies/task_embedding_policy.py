"""Policy class for Task Embedding envs."""
import abc

import akro

from garage.tf.policies.policy import StochasticPolicy


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
                shape :math:`(Z, )`. Z is the dimension of the latent
                embedding.
            dict: Embedding distribution information.

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
            dict: Action distribution information.

        """

    @abc.abstractmethod
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
            dict: Action distribution information.

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
            dict: Action distribution information.

        """

    @abc.abstractmethod
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
            dict: Action distribution information.

        """

    @abc.abstractmethod
    def get_action_given_latent(self, observation, latent):
        """Sample an action given observation and latent.

        Args:
            observation (np.ndarray): Observation from the environment,
                with shape :math:`(O, )`. O is the dimension of observation.
            latent (np.ndarray): Latent, with shape :math:`(Z, )`. Z is the
                dimension of latent embedding.

        Returns:
            np.ndarray: Action sampled from the policy,
                with shape :math:`(A, )`. A is the dimension of action.
            dict: Action distribution information.

        """

    @abc.abstractmethod
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
            dict: Action distribution information.

        """

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

"""Policy class for multitask envs."""
import abc

from garage.tf.embeddings import Embedding
from garage.tf.embeddings import StochasticEmbedding
from garage.tf.embeddings.utils import concat_spaces
from garage.tf.policies import Policy
from garage.tf.policies import StochasticPolicy


class MultitaskPolicy(Policy):
    """Base class for multitask policies in TensorFlow.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        embedding (Embedding): A embedding network.
        task_space (akro.Box): A space for task.

    """
    def __init__(self, name, env_spec, embedding, task_space):
        super().__init__(name, env_spec)
        self._embedding = embedding
        self._task_space = task_space
        self._task_observation_space = concat_spaces(
            self._task_space, self._env_spec.observation_space)

    @abc.abstractmethod
    def get_action_from_onehot(self, observation, onehot):
        """Get action sampled from the policy based on onehot index.

        Args:
            observation (np.ndarray): Observation from the environment.
            onehot (np.ndarray): One hot task index.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions_from_onehots(self, observations, onehots):
        """Get actions sampled from the policy based on onehot indices.

        Args:
            observations (np.ndarray): Observations from the environment.
            onehots (np.ndarray): One hot task indices.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_action_from_latent(self, observation, latent):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.
            latent (np.ndarray): Latent.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions_from_latents(self, observations, latents):
        """Get actions sampled from the policy.

        Args:
            observations (np.ndarray): Observations from the environment.
            latents (np.ndarray): Latent.

        Returns:
            (np.ndarray): Actions sampled from the policy.

        """

    def get_latent(self, onehot):
        return self._embedding.get_latent(onehot)

    @property
    def embedding(self):
        return self._embedding

    @property
    def latent_space(self):
        return self._embedding.latent_space

    @property
    def embedding_spec(self):
        return self._embedding.embedding_spec

    @property
    def task_space(self):
        return self._task_space

    @property
    def task_observation_space(self):
        return self._task_observation_space

    def split_observation(self, observation):
        """Splits up observation into task onehot and vanilla environment observation.

        Args:
            observation (np.ndarray): task onehot concatenated with vanilla
                environment observation.

        Returns:
            (tuple): task onehot, vanilla environment observation
        """
        return observation[:self.task_space.flat_dim], observation[
            self.task_space.flat_dim:]

    def get_trainable_vars(self):
        """Get trainable variables.

        The trainable vars of a multitask policy should be the trainable vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._variable_scope.trainable_variables() + self._embedding.get_trainable_vars()

    def get_global_vars(self):
        """Get global variables.

        The global vars of a multitask policy should be the global vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return self._variable_scope.global_variables() + self._embedding.get_global_vars()


class StochasticMultitaskPolicy(StochasticPolicy, MultitaskPolicy):
    """StochasticMultitaskPolicy."""

    def __init__(self, env_spec, embedding: StochasticEmbedding,
                 task_space, name='StochasticMultitaskPolicy'):
        super().__init__(name, env_spec, embedding, task_space)

    @property
    def embedding_distribution(self):
        """Embedding distribution."""
        return self._embedding.distribution

    def embedding_dist_info_sym(self, obs_var, state_info_vars):
        """Return the symbolic distribution information about the embedding.

        Args:
            obs_var(tf.Tensor): symbolic variable for observations
            state_info_vars(dict): a dictionary whose values should contain information
                about the state of the policy at the time it received the observation.

        """
        return self._embedding.dist_info_sym(obs_var, state_info_vars)

    def embedding_dist_info(self, obs, state_infos):
        """Return the distribution information about the embedding.

        Args:
            obs_var(tf.Tensor): observation values
            state_info_vars(dict): a dictionary whose values should contain information
                about the state of the policy at the time it received the observation.

        """

    @property
    def action_distribution(self):
        """Action distribution."""

    def action_dist_info_sym(self, obs_var, state_info_vars):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.

        Args:
            obs_var (tf.Tensor): symbolic variable for observations
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        """

    def action_dist_info(self, obs, state_infos):
        """Distribution info.

        Return the distribution information about the actions.

        Args:
            obs (tf.Tensor): observation values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation

        """

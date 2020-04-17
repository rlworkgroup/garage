"""Base class for embedding networks in TensorFlow."""
import abc

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class EmbeddingSpec:
    """Specification of an embedding network.

    Args:
        input_space (akro.Space): The input space of the embedding.
        latent_space (akro.Space): The latent space of the embedding.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, input_space, latent_space):
        self.input_space = input_space
        self.latent_space = latent_space


class Embedding(abc.ABC):
    """An embedding.

    An embedding is a mapping from a high-dimensional vector to a
    low-dimensional one. In other words, it represents a concise representation
    of a high-dimensional space.

    Args:
        name (str): The name of this embedding.
        embedding_spec (EmbeddingSpec): The specification of this embedding.

    """

    def __init__(self, name, embedding_spec):
        self._name = name
        self._embedding_spec = embedding_spec
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

    @abc.abstractmethod
    def get_latent(self, given):
        """Get latent given input.

        Args:
            given (np.ndarray): Input to be embedded.

        Returns:
            (np.ndarray): The embedding of input.

        """

    @abc.abstractmethod
    def get_latents(self, givens):
        """Get latents given an array of inputs.

        Args:
            givens (np.ndarray): An array of inputs to be embedded.

        Returns:
            (np.ndarray): The embeddings of inputs.

        """

    def reset(self):
        """Reset the embedding."""

    @property
    def vectorized(self):
        """Boolean for vectorized.

        Returns:
            bool: Indicates whether the embedding is vectorized. If True, it
                should implement get_latents(), and support resetting with
                multiple simultaneous states.

        """

    @property
    def name(self):
        """str: The name of the embedding."""
        return self._name

    @property
    def input_space(self):
        """akro.Space: The input space of the embedding."""
        return self._embedding_spec.input_space

    @property
    def latent_space(self):
        """akro.Space: The latent space of the embedding."""
        return self._embedding_spec.latent_space

    @property
    def embedding_spec(self):
        """EmbeddingSpec: The specification of the embedding."""
        return self._embedding_spec

    @property
    def recurrent(self):
        """Whether the embedding uses recurrent network or not.

        Returns:
            bool: Indicating if the embedding is recurrent.

        """
        return False

    @property
    def state_info_keys(self):
        """State info keys.

        Returns:
            List[str]: keys for the information related to the embedding's
                state when taking an action.

        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """State info specifcation.

        Returns:
            List[str]: keys and shapes for the information related to the
                embedding's state when taking an action.

        """
        return list()

    def get_trainable_vars(self):
        """Get trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get global variables.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return self._variable_scope.global_variables()

    def get_params(self):
        """Get the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self.get_trainable_vars()

    def get_param_shapes(self):
        """Get parameter shapes.

        Returns:
            List[tuple]: A list of variable shapes.

        """
        if self._cached_param_shapes is None:
            params = self.get_params()
            param_values = tf.compat.v1.get_default_session().run(params)
            self._cached_param_shapes = [val.shape for val in param_values]
        return self._cached_param_shapes

    def get_param_values(self):
        """Get param values.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        params = self.get_params()
        param_values = tf.compat.v1.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.

        """
        param_values = unflatten_tensors(param_values, self.get_param_shapes())
        for param, value in zip(self.get_params(), param_values):
            param.load(value)

    def flat_to_params(self, flattened_params):
        """Unflatten tensors according to their respective shapes.

        Args:
            flattened_params (np.ndarray): A numpy array of flattened params.

        Returns:
            List[np.ndarray]: A list of parameters reshaped to the
                shapes specified.

        """
        return unflatten_tensors(flattened_params, self.get_param_shapes())

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_cached_params']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self._cached_params = None
        self.__dict__.update(state)


class StochasticEmbedding(Embedding):
    """An stochastic embedding.

    An stochastic embedding maps an input to a distribution, but not a
    deterministic vector.

    """

    @property
    @abc.abstractmethod
    def distribution(self):
        """Distribution."""

    @abc.abstractmethod
    def dist_info_sym(self, in_var, state_info_vars, name):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.

        Args:
            in_var (tf.Tensor): symbolic variable for inputs
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the embedding at the time it
                received the observation.
            name (str): Name for symbolic graph.

        """

    def dist_info(self, an_input, state_infos):
        """Distribution info.

        Return the distribution information about the actions.

        Args:
            an_input (tf.Tensor): input values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the embedding at the time it
                received the observation

        """

"""Base class for embedding networks in TensorFlow."""
import abc


class Embedding(abc.ABC):
    def __init__(self, name, embedding_spec):
        self._name = name
        self._embedding_spec = embedding_spec
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

    @abc.abstractmethod
    def get_latent(self, given):
        """Get latent sampled from the embedding.

        Args:
            given (np.ndarray): Input from the environment.

        Returns:
            (np.ndarray): Latent sampled from the policy.

        """

    @abc.abstractmethod
    def get_latents(self, givens):
        """Get latents sampled from the embedding.

        Args:
            givens (np.ndarray): Inputs from the environment.

        Returns:
            (np.ndarray): Latents sampled from the policy.

        """

    def reset(self):
        """Reset the embedding."""
        pass

    @property
    def vectorized(self):
        """Boolean for vectorized.

        Returns:
            bool: Indicates whether the embedding is vectorized. If True, it
                should implement get_latents(), and support resetting with
                multiple simultaneous states.

        """
        return False

    @property
    def name(self):
        return self._name

    @property
    def input_space(self):
        return self._embedding_spec.input_space

    @property
    def latent_space(self):
        return self._embedding_spec.latent_space

    @property
    def embedding_spec(self):
        return self._embedding_spec

    @property
    def recurrent(self):
        """Whether the embedding uses recurrent network or not.

        Returns:
            bool: Indicating if the policy is recurrent.

        """
        return False

    def log_diagnostics(self):
        """Log extra information per iteration based on the collected paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """
        pass

    @property
    def state_info_keys(self):
        """State info keys.

        Returns:
            List[str]: keys for the information related to the policy's state
                when taking an action.

        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """State info specifcation.

        Returns:
            List[str]: keys and shapes for the information related to the
                policy's state when taking an action.

        """
        return list()

    def terminate(self):
        """Clean up operation."""
        pass

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
        if self._cached_params is None:
            self._cached_params = self.get_trainable_vars()
        return self._cached_params

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
    @property
    @abc.abstractmethod
    def distribution(self):
        """Distribution."""
    
    @abc.abstractmethod
    def dist_info_sym(self, in_var, state_info_vars):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.

        Args:
            in_var (tf.Tensor): symbolic variable for inputs
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.

        """
        raise NotImplementedError

    def dist_info(self, an_input, state_infos):
        """Distribution info.

        Return the distribution information about the actions.

        Args:
            an_input (tf.Tensor): input values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation

        """

"""Base class for policies in TensorFlow."""
import abc

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class Policy(abc.ABC):
    """Base class for policies in TensorFlow.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, name, env_spec):
        self._name = name
        self._env_spec = env_spec
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get action sampled from the policy.

        Args:
            observations (list[np.ndarray]): Observations from the environment.

        Returns:
            (np.ndarray): Actions sampled from the policy.

        """

    def reset(self, dones=None):
        """Reset the policy.

        If dones is None, it will be by default np.array([True]) which implies
        the policy will not be "vectorized", i.e. number of parallel
        environments for training data sampling = 1.

        Args:
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """

    @property
    def name(self):
        """Name of the policy model.

        Returns:
            str: Policy name. This is also the variable scope.

        """
        return self._name

    @property
    def vectorized(self):
        """Boolean for vectorized.

        Returns:
            bool: Indicates whether the policy is vectorized. If True, it
                should implement get_actions(), and support resetting with
                multiple simultaneous states.

        """
        return False

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Space: The observation space of the environment.

        """
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Action space.

        Returns:
            akro.Space: The action space of the environment.

        """
        return self._env_spec.action_space

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    @property
    def recurrent(self):
        """Whether the policy uses recurrent network or not.

        Returns:
            bool: Indicating if the policy is recurrent.

        """
        return False

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """

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


class StochasticPolicy(Policy):
    """StochasticPolicy."""

    @property
    @abc.abstractmethod
    def distribution(self):
        """Distribution."""

    @abc.abstractmethod
    def dist_info_sym(self, obs_var, state_info_vars, name='dist_info_sym'):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.

        Args:
            obs_var (tf.Tensor): symbolic variable for observations
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        """

    def dist_info(self, obs, state_infos):
        """Distribution info.

        Return the distribution information about the actions.

        Args:
            obs (tf.Tensor): observation values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation

        """

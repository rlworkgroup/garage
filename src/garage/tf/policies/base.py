"""Base class for Policies."""
import abc

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class Policy(abc.ABC):
    """Base class for Policies.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, name, env_spec):
        self._name = name
        self._env_spec = env_spec
        self._variable_scope = None
        self._cached_params = {}
        self._cached_param_shapes = {}

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
        """str: Name of the policy model and the variable scope."""
        return self._name

    @property
    def vectorized(self):
        """Boolean for vectorized.

        Returns:
            bool: Indicates whether the policy is vectorized. If True, it
            should implement get_actions(), and support resetting with multiple
            simultaneous states.

        """
        return False

    @property
    def observation_space(self):
        """akro.Space: The observation space of the environment."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """akro.Space: The action space for the environment."""
        return self._env_spec.action_space

    @property
    def env_spec(self):
        """garage.EnvSpec: Policy environment specification."""
        return self._env_spec

    @property
    def recurrent(self):
        """bool: Indicating if the policy is recurrent."""
        return False

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths."""

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

    def get_params(self, trainable=True):
        """Get the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
            variable scope.

        """
        return self.get_trainable_vars()

    def get_param_shapes(self, **tags):
        """Get parameter shapes."""
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.compat.v1.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [
                val.shape for val in param_values
            ]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        """Get param values.

        Args:
            tags (dict): A map of parameters for which the values are required.
        Returns:
            param_values (np.ndarray): Values of the parameters evaluated in
            the current session

        """
        params = self.get_params(**tags)
        param_values = tf.compat.v1.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values, name=None, **tags):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.
            tags (dict): A map of parameters for which the values should be
            loaded.
        """
        param_values = unflatten_tensors(param_values,
                                         self.get_param_shapes(**tags))
        for param, value in zip(self.get_params(**tags), param_values):
            param.load(value)

    def flat_to_params(self, flattened_params, **tags):
        """Unflatten tensors according to their respective shapes.

        Args:
            flattened_params (np.ndarray): A numpy array of flattened params.
            tags (dict): A map specifying the parameters and their shapes.

        Returns:
            tensors (List[np.ndarray]): A list of parameters reshaped to the
            shapes specified.

        """
        return unflatten_tensors(flattened_params,
                                 self.get_param_shapes(**tags))

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_cached_params']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self._cached_params = {}
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

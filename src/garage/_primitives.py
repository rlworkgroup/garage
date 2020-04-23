"""Policy interface."""
import abc


class Primitive(abc.ABC):
    """Primitive interface.

    Args:
        name (str): Name for logs, variable scopes, etc.
        spec (garage.InOutSpec): Input/output specification this primitive will
            satisfy.

    """

    def __init__(self, name, spec):
        self._name = name
        self._spec = spec

    @abc.abstractmethod
    def get_parameters(self):
        """Get parameters of this primitive.

        Returns:
            dict[str, np.ndarray]: A dictionary mapping from names to values of
                each parameter.

        """

    @abc.abstractmethod
    def set_parameters(self, parameters):
        """Set the parameters of this primtive.

        Args:
            parameters (dict[str, np.ndarray]): A dictionary mapping from names
                to values of each parameter.

        """

    @property
    def spec(self):
        """InOutSpec: The input/output spec for this primitive."""
        return self._spec

    @property
    def name(self):
        """str: Name of this primitive."""
        return self._name

    @abc.abstractmethod
    def clone(self, name):
        """Create a fully-independent copy of this primitive.

        Args:
            name (str): Name for the cloned instance.

        Returns:
            Primitive: A fully-independent copy of this primitive.

        """

    @abc.abstractmethod
    def __getstate__(self):
        """See Object.__getstate__.

        Returns:
            dict: Pickling state for the primitive

        """

    @abc.abstractmethod
    def __setstate__(self, state):
        """See Object.__setstate__.

        Args:
            state (dict): Unpickled state of this primitive.

        """


class Policy(Primitive):
    """Policy interface.

    Args:
        name (str): Name for logs, variable scopes, etc.
        env_spec (garage.envs.EnvSpec): Environment spec this Policy
            will satisfy.

    """

    def __init__(self, name, env_spec):
        super().__init__(name, env_spec.transpose())

    @abc.abstractmethod
    def get_action(self, observation, policy_state=None):
        r"""Get action conditioned on observation.

        Args:
            observation (np.ndarray): Observation from an environment,
                conforming to :obj:`env_spec.observation_space`. Shape is
                :math:`(O*,)`.
            policy_state (object): State information.

        Returns:
            np.ndarray: An action, conditioned on the observation, conforming
                to :obj:`env_spec.action_space`. Shape is :math:`(A^*,)`.
            object: State information. If not None, this should be passed to
                the next call to `get_action`.

        """

    @abc.abstractmethod
    def get_actions(self, observations, policy_states=None):
        r"""Get actions conditioned on observations.

        Note:
            This API is assumed to be vectorized, unless otherwise noted.

        Args:
            observations (np.ndarray): Array of observations of shape
                :math:`(N, O^*)` containing observations from the environment.
                These should conform to :obj:`env_spec.observation_space`.
            policy_states (object): State information.

        Returns:
            np.ndarray: An array of actions of shape :math:`(N, A*)`,
                conforming to :obj:`env_spec.action_space`.
            object: State information. If not None, this should be passed to
                the next call to `get_actions`.

        """

    @abc.abstractmethod
    def get_reset_state(self):
        """Get the initial (reset) value of policy_state.

        Returns:
            object: Policy state information, but be used with get_action.

        """

    @abc.abstractmethod
    def get_reset_states(self, batch_size):
        """Get the reset value of policy_states for a given batch size.

        Args:
            batch_size (int): Observation/action batch size for the calls to
               `get_actions` this state will be used with.

        Returns:
            object: Policy state representing a newly-reset policy.

        """

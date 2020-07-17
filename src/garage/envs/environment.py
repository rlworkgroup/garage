"""Base Garage Environment API."""

import abc


class Environment(abc.ABC):
    """The main API for garage environments.

    The main public API methods are:

        step
        reset
        render
        visualize
        close

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `pickle` or `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__getstate__()` and
        `__setstate()__` to add your custom pickle logic.

    Note: a default reward range set to [-inf,+inf] already exists. Set it if
        you want a narrower range.

    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    @abc.abstractmethod
    def reset(self):
        """Resets the environment and returns the first `TimeStep` of sequence.

        Returns:
            garage.TimeStep: The first TimeStep. Attributes:
                * env_spec (garage.envs.EnvSpec): Environment specification.
                observation (None): there is no previous observation.
                * action (None): no action is given when reset.
                * reward (None): at reset, the reward is undefined.
                * next_observation: an `np.ndarray`, or a nested dict, list or
                    tuple corresponding to :obj:`env_spec.observation_space`.
                    The initial observation.
                * terminal (bool): The termination signal for the this time
                    step.
                * env_info (dict): A dict arbitrary environment state
                    information.
                * agent_info (numpy.ndarray): A dict of arbitrary agent
                    state information.
                * step_type (garage.StepType): a `StepType` enum of value
                    StepType.FIRST.

        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment using to the action and returns a `TimeStep`.

        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and
        `action` will be ignored.

        This method will also start a new sequence if called after the
        environment has been constructed and `reset` has not been called.
        Again, in this case `action` will be ignored.

        Args:
            action (np.ndarray): A NumPy array, or a nested dict, list or tuple
                of arrays corresponding to `action_spec()`.

        Returns:
            garage.TimeStep: The TimeStep after the action. Attributes:
                * env_spec (garage.envs.EnvSpec): Environment specification.
                * observation: an `np.ndarray`, or a nested dict, list or
                    tuple corresponding to :obj:`env_spec.observation_space`.
                    The observation before applying the action.
                * action (np.ndarray): A NumPy array, or a nested dict, list or
                    tuple of arrays corresponding to `action_spec()`.
                * reward (float): The reward for taking the action given the
                    observation, at the this time step.
                * next_observation: an `np.ndarray`, or a nested dict, list or
                    tuple corresponding to :obj:`env_spec.observation_space`.
                    The observation after applying the action.
                * terminal (bool): The termination signal for the this time
                    step.
                * env_info (dict): A dict arbitrary environment state
                    information.
                * agent_info (numpy.ndarray): A dict of arbitrary agent
                    state information.
                * step_type (garage.StepType): a `StepType` enum value. Can
                    either be StepType.FIRST, StepType.MID, StepType.LAST,
                    or StepType.TIMEOUT.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        * ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Make sure that your class's metadata 'render.modes' key includes
        the list of supported modes.

        Args:
            mode (str): the mode to render with. The string must be present in
            self.metadata['render.modes'].

        Raises:
            NotImplementedError: If the mode is not implemented.

        For example:

        class MyEnv(Env):
            metadata = {'render.modes': ['display', 'rgb_array']}

            def render(self, mode):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'ansi':
                    ... # return text output
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an
                    exception
        """
        raise NotImplementedError

    @abc.abstractmethod
    def visualize(self):
        """Displays the environment visualization."""

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def __str__(self):
        """Returns a string representation of the environment.

        Returns:
            str: String representation of the environment.
        """
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment.

        Returns:
            self: the class object itself.
        """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment.

        Args:
            args: input arguments.
        """
        self.close()

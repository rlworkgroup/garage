"""Base Garage Environment API."""

import abc


class Environment(abc.ABC):
    """The main API for garage environments.

    The public API methods are:

    +-----------------------+
    | Functions             |
    +=======================+
    | reset()               |
    +-----------------------+
    | step()                |
    +-----------------------+
    | render()              |
    +-----------------------+
    | visualize()           |
    +-----------------------+
    | close()               |
    +-----------------------+

    Set the following properties:

    +-----------------------+-------------------------------------------------+
    | Properties            | Description                                     |
    +=======================+=================================================+
    | action_space          | The action space specification                  |
    +-----------------------+-------------------------------------------------+
    | observation_space     | The observation space specification             |
    +-----------------------+-------------------------------------------------+
    | spec                  | The environment specifications                  |
    +-----------------------+-------------------------------------------------+
    | render_modes          | The list of supported render modes              |
    +-----------------------+-------------------------------------------------+

    Example of a simple rollout loop:

    .. code-block:: python

        env = MyEnv()
        policy = MyPolicy()
        first_observation, episode_info = env.reset()
        env.visualize()  # visualization window opened

        episode = []
        # Determine the first action
        first_action = policy.get_action(first_observation, episode_info)
        episode.append(env.step(first_action))

        while not episode[-1].last():
           action = policy.get_action(episode[-1].next_observation)
           episode.append(env.step(action))

        env.close()  # visualization window closed

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__setstate__()` and
        `__getstate__()` to add your custom pickle logic.

        You might want to refer to the EzPickle module:
        https://github.com/openai/gym/blob/master/gym/utils/ezpickle.py
        for a lightweight way of pickle and unpickle via constructor
        arguments.

    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """akro.Space: The action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """akro.Space: The observation space specification."""

    @property
    @abc.abstractmethod
    def spec(self):
        """EnvSpec: The environment specification."""

    @property
    @abc.abstractmethod
    def render_modes(self):
        """list: A list of string representing the supported render modes.

        See render() for a list of modes.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the environment.

        Returns:
            numpy.ndarray: The first observation. It must conforms to
            `observation_space`.
            dict: The episode-level information. Note that this is not part
            of `env_info` provided in `step()`. It contains information of
            the entire episode， which could be needed to determine the first
            action (e.g. in the case of goal-conditioned or MTRL.)

        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment with the action and returns a `TimeStep`.

        If the environment returned the last `TimeStep` of a sequence (either
        of type TERMINAL or TIMEOUT) at the previous step, this call to
        `step()` will start a new sequence and `action` will be ignored.

        If `spec.max_episode_length` is reached after applying the action
        and the environment has not terminated the episode, `step()` should
        return a `TimeStep` with `step_type==StepType.TIMEOUT`.

        If possible, update the visualization display as well.

        Args:
            action (object): A NumPy array, or a nested dict, list or tuple
                of arrays conforming to `action_space`.

        Returns:
            TimeStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an `numpy.ndarray` with shape (x, y, 3) and type
            uint8, representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        * ansi: Return a string (str) or `StringIO.StringIO` containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).

        Make sure that your class's `render_modes` includes the list of
        supported modes.

        For example:

        .. code-block:: python

            class MyEnv(Environment):
                def render_modes(self):
                    return ['rgb_array', 'ansi']

                def render(self, mode):
                    if mode == 'rgb_array':
                        return np.array(...)  # return RGB frame for video
                    elif mode == 'ansi':
                        ...  # return text output
                    else:
                        raise ValueError('Supported render modes are {}, but '
                                         'got render mode {} instead.'.format(
                                             self.render_modes, mode))

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.

        This function should be called **only once** after `reset()` to set up
        the visualization display. The visualization should be updated
        when the environment is changed (i.e. when `step()` is called.)

        Calling `close()` will deallocate any resources and close any
        windows created by `visualize()`. If `close()` is not explicitly
        called, the visualization will be closed when the environment is
        destructed (i.e. garbage collected).

        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        This method should close all windows invoked by `visualize()`.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically `close()` themselves when they are
        garbage collected or when the program exits.
        """

    def __del__(self):
        """Environment destructor."""
        self.close()

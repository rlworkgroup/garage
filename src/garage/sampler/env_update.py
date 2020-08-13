"""A callable that "updates" an environment."""
import warnings


class EnvUpdate:
    """A callable that "updates" an environment.

    Implementors of this interface can be called on environments to update
    them. The passed in environment should then be ignored, and the returned
    one used instead.

    Since no new environment needs to be passed in, this type can also
    be used to construct new environments.

    """

    # pylint: disable=too-few-public-methods

    def __call__(self, old_env=None):
        """Update an environment.

        Note that this implementation does nothing.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        return old_env


class NewEnvUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that creates a new environment every update.

    Args:
        env_constructor (Callable[Environment]): Callable that constructs an
            environment.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, env_constructor):
        self._env_constructor = env_constructor

    def __call__(self, old_env=None):
        """Update an environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        if old_env:
            old_env.close()
        return self._env_constructor()


class SetTaskUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that calls set_task with the provided task.

    Args:
        env_constructor (Callable[Environment]): Callable that constructs an
            environment.
        task (object): Opaque task type.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, env_constructor, task):
        self._env_constructor = env_constructor
        self._task = task

    def __call__(self, old_env=None):
        """Update an environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        if old_env is None:
            old_env = self._env_constructor()
        old_env.set_task(self._task)
        return old_env


class ExistingEnvUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that carries an already constructed environment.

    Args:
        env (Environment): The environment.

    """

    def __init__(self, env):
        self._env = env

    def __call__(self, old_env=None):
        """Update an environment.

        This implementation does not close the old environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        return self._env

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        warnings.warn('ExistingEnvUpdate is generally not the most efficient '
                      'method of transmitting environments to other '
                      'processes.')
        return self.__dict__

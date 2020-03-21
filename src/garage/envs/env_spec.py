"""EnvSpec class."""

from garage import InOutSpec


class EnvSpec(InOutSpec):
    """Describes the action and observation spaces of an environment.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.

    """

    def __init__(self, observation_space, action_space):
        super().__init__(action_space, observation_space)

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space

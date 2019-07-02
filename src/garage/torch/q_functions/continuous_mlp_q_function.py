"""This modules creates a continuous MLP Q fcuntion network."""


class ContinuousMLPQFunction:
    """
    This class implements a q value network.

    It predicts the q value for all actions based on the input state. It uses
    an MLP to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 nn_module,
                 name='ContinuousMLPQFunction',
                 input_include_goal=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            nn_module():
                A PyTorch module.
            name(str, optional):
                A str contains the name of the policy.
        """
        self._env_spec = env_spec
        self._nn_module = nn_module
        self._name = name
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high

    def get_qval(self, observations):
        """Return Q value."""
        return self._nn_module.forward(observations)

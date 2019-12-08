"""Base class for all baselines."""
import abc


class Baseline(abc.ABC):
    """Base class for all baselines.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, env_spec):
        self._mdp_spec = env_spec

    @abc.abstractmethod
    def get_param_values(self):
        """Get parameter values.

        Returns:
            List[np.ndarray]: A list of values of each parameter.

        """

    @abc.abstractmethod
    def set_param_values(self, flattened_params):
        """Set param values.

        Args:
            flattened_params (np.ndarray): A numpy array of parameter values.

        """

    @abc.abstractmethod
    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """

    @abc.abstractmethod
    def predict(self, path):
        """Predict value based on paths.

        Args:
            path (dict[numpy.ndarray]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """

    def log_diagnostics(self, paths):
        """Log diagnostic information.

        Args:
            paths (list[dict]): A list of collected paths.

        """

import abc


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.
    """

    @abc.abstractmethod
    def train_once(self, itr, paths):
        """Performs one step of policy optimization
        given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.
        """
        pass

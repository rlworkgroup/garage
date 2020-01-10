"""Interface of RLAlgorithm."""
import abc

from dowel import tabular
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.

    Note:
        If sampler_cls isn't specified to the LocalRunner,
        self.sampler_cls is required to provide default sampler
        for algorithm.

    """

    @abc.abstractmethod
    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle or None.

        """

    def evaluate_performance(self, itr, batch):
        # pylint: disable=no-self-use
        r"""Evaluate the performance of the algorithm.

        Args:
            itr (int): Iteration number.
            batch (dict): Evaluation trajectories, representing
                the best current performance of the algorithm, with keys:
                * env_spec (garage.envs.EnvSpec): Specification for the
                environment from which this data was sampled.
                * observations (numpy.ndarray): A numpy array containing the
                    observations for all time steps in this batch.
                * actions (numpy.ndarray): A  numpy array containing the
                    actions for all time steps in this batch.
                * rewards (numpy.ndarray): A numpy array containing the
                    rewards for all time steps in this batch.
                * terminals (numpy.ndarray): A boolean numpy array
                    containing the termination signals for all time steps
                    in this batch.
                * env_infos (dict): A dict of numpy arrays arbitrary
                    environment state information.
                * agent_infos (numpy.ndarray): A dict of numpy arrays
                    arbitrary agent state information.
                * lengths (numpy.ndarray): An integer numpy array
                    containing the length of each trajectory in this batch.
                * discount (float): Discount value, from algorithm's property.

        Returns:
            numpy.ndarray: Undiscounted returns.

        """
        returns = []
        for reward in batch['rewards']:
            rtn = np_tensor_utils.discount_cumsum(reward, batch['discount'])
            returns.append(rtn)

        average_discounted_return = np.mean([rtn[0] for rtn in returns])

        undiscounted_returns = [sum(reward) for reward in batch['rewards']]

        tabular.record('Iteration', itr)
        tabular.record('Evaluation/NumTrajs', len(returns))

        tabular.record('Evaluation/AverageDiscountedReturn',
                       average_discounted_return)
        tabular.record('Evaluation/AverageReturn',
                       np.mean(undiscounted_returns))
        tabular.record('Evaluation/StdReturn', np.std(undiscounted_returns))
        tabular.record('Evaluation/MaxReturn', np.max(undiscounted_returns))
        tabular.record('Evaluation/MinReturn', np.min(undiscounted_returns))

        return undiscounted_returns

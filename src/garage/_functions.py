"""Functions exposed directly in the garage namespace."""
from collections import defaultdict

from dowel import tabular
import numpy as np

import garage
from garage.misc.tensor_utils import discount_cumsum


def log_multitask_performance(itr, batch, discount, task_names=None):
    """Log per-task performance and average performance.

    Args:
        itr (int): Iteration.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.
        task_names (dict[str]): Names of all tasks to evaluate.
            In case when a task in task_names does not exist in the give batch,
            log NaN as a placeholder.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    traj_by_name = defaultdict(list)
    for trajectory in batch.split():
        try:
            task_name = trajectory.env_infos['task_name'][0]
        except KeyError:
            task_name = trajectory.env_infos['task_id'][0]
        traj_by_name[task_name].append(trajectory)
    if task_names is None:
        for (task_name, trajectories) in traj_by_name.items():
            log_performance(itr,
                            garage.TrajectoryBatch.concatenate(*trajectories),
                            discount,
                            prefix=task_name)
    else:
        for task_name in sorted(task_names):
            if task_name in traj_by_name:
                trajectories = traj_by_name[task_name]
                log_performance(
                    itr,
                    garage.TrajectoryBatch.concatenate(*trajectories),
                    discount,
                    prefix=task_name)
            else:
                with tabular.prefix(task_name + '/'):
                    tabular.record('Iteration', np.nan)
                    tabular.record('NumTrajs', np.nan)
                    tabular.record('AverageDiscountedReturn', np.nan)
                    tabular.record('AverageReturn', np.nan)
                    tabular.record('StdReturn', np.nan)
                    tabular.record('MaxReturn', np.nan)
                    tabular.record('MinReturn', np.nan)
                    tabular.record('CompletionRate', np.nan)
                    # pylint: disable=fixme
                    # TODO: Always logging SuccessRate is problematic,
                    #   because this environment may not have SuccessRate.
                    #   This issue cannot be fixed here because no actual data
                    #   is accessible in this context.
                    #   Maybe fix CsvOutput to allow inconsistent field names.
                    tabular.record('SuccessRate', np.nan)

    return log_performance(itr, batch, discount=discount, prefix='Average')


def log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    completion = []
    success = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))
        completion.append(float(trajectory.terminals.any()))
        if 'success' in trajectory.env_infos:
            success.append(float(trajectory.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumTrajs', len(returns))

        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('CompletionRate', np.mean(completion))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns

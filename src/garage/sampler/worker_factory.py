"""Worker factory used by Samplers to construct Workers."""
import psutil

from garage.experiment.deterministic import get_seed
from garage.sampler.default_worker import DefaultWorker


def identity_function(value):
    """Do nothing.

    This function exists so it can be pickled.

    Args:
        value(object): A value.

    Returns:
        object: The value.

    """
    return value


class WorkerFactory:
    """Constructs workers for Samplers.

    The intent is that this object should be sufficient to avoid subclassing
    the sampler. Instead of subclassing the sampler for e.g. a specific
    backend, implement a specialized WorkerFactory (or specify appropriate
    functions to this one). Not that this object must be picklable, since it
    may be passed to workers. However, its fields individually need not be.

    All arguments to this type must be passed by keyword.

    Args:
        max_episode_length(int): The maximum length episodes which will be
            sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.

    """

    def __init__(
            self,
            *,  # Require passing by keyword.
            max_episode_length,
            is_tf_worker=False,
            seed=get_seed(),
            n_workers=psutil.cpu_count(logical=False),
            worker_class=DefaultWorker,
            worker_args=None):
        self.n_workers = n_workers
        self._seed = seed
        self._max_episode_length = max_episode_length
        if is_tf_worker:
            # Import here to avoid hard dependency on TF.
            # pylint: disable=import-outside-toplevel
            from garage.tf.samplers import TFWorkerClassWrapper
            worker_class = TFWorkerClassWrapper(worker_class)
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args

    def prepare_worker_messages(self, objs, preprocess=identity_function):
        """Take an argument and canonicalize it into a list for all workers.

        This helper function is used to handle arguments in the sampler API
        which may (optionally) be lists. Specifically, these are agent, env,
        agent_update, and env_update. Checks that the number of parameters is
        correct.

        Args:
            objs(object or list): Must be either a single object or a list
                of length n_workers.
            preprocess(function): Function to call on each single object before
                creating the list.

        Raises:
            ValueError: If a list is passed of a length other than `n_workers`.

        Returns:
            List[object]: A list of length self.n_workers.

        """
        if isinstance(objs, list):
            if len(objs) != self.n_workers:
                raise ValueError(
                    'Length of list doesn\'t match number of workers')
            return [preprocess(obj) for obj in objs]
        else:
            return [preprocess(objs) for _ in range(self.n_workers)]

    def __call__(self, worker_number):
        """Construct a worker given its number.

        Args:
            worker_number(int): The worker number. Should be at least 0 and
                less than or equal to `n_workers`.

        Raises:
            ValueError: If the worker number is greater than `n_workers`.

        Returns:
            garage.sampler.Worker: The constructed worker.

        """
        if worker_number >= self.n_workers:
            raise ValueError('Worker number is too big')
        return self._worker_class(worker_number=worker_number,
                                  seed=self._seed,
                                  max_episode_length=self._max_episode_length,
                                  **self._worker_args)

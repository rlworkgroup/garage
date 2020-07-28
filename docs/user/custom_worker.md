# Change how your algorithm samples (Implement a Custom Worker)

```eval_rst
In sampling, :code:`Worker` perform rollouts to get samples. In this tutorial,
we will implement a custom worker to meet the requirements of the RL2
algorithm :cite:`duan2016rl`.
```

## `Worker` interface and `DefaultWorker`

In Garage, all `Worker`s need to implement the [`Worker` interface](https://garage.readthedocs.io/en/latest/_autoapi/garage/sampler/index.html#garage.sampler.Worker).
Garage provides a [`DefaultWorker`](https://garage.readthedocs.io/en/latest/_autoapi/garage/sampler/index.html#garage.sampler.DefaultWorker)
for general algorithms. Below is the `rollout()` method of `DefaultWorker`, the
most important method of `Worker`.

```py
    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout():
            pass
        return self.collect_rollout()
```

To implement a custom worker, an easy way is to override the methods we want to
change (e.g. `rollout()`) instead of implementing our custom worker from
scratch. For example:

```py
from garage.sampler import DefaultWorker

class MyWorker(DefaultWorker):
    """Initialize a worker.

    Args:
        seed(int): The seed to use to initialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number):
        ...
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        ...

    def rollout(self):
        ...
```

## Custom Worker for RL2

In RL2, different from general RL algorithms, multiple trajectories are sampled
in one trial/meta batch. Thus we can move the original rollout into a loop:

```py
from garage.sampler import DefaultWorker

class RL2Worker(DefaultWorker):
    """Initialize a worker for RL2.

    Args:
        seed(int): The seed to use to initialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_paths_per_trial (int): Number of trajectories sampled per trial/
            meta batch. Policy resets in the beginning of a meta batch,
            and obtain `n_paths_per_trial` trajectories in one meta batch.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number,
            n_paths_per_trial=2):
        self._n_paths_per_trial = n_paths_per_trial
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def rollout(self):
        for _ in range(self._n_paths_per_trial):
            self.start_rollout()
            while not self.step_rollout():
                pass
        return self.collect_rollout()
```

This is not enough, because in RL2 the agent/policy only resets at the beginning
of a trail/meta batch. So we need to override `start_rollout()`. And we want to
record the batch index.

```py
...
    def start_rollout(self):
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def rollout(self):
        self.agent.reset()
        for _ in range(self._n_paths_per_trial):
            self.start_rollout()
            while not self.step_rollout():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._rewards),
                                                 self._worker_number)
        return self.collect_rollout()
```

We have completed our custom worker for RL2. For reference, you can see the
complete implementation of RL2 algorithm and its worker [here](https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2.py).

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----
*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

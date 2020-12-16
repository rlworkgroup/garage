# Change How Your Algorithm Samples (Implement a Custom Worker)

```eval_rst
In sampling, :code:`Worker` perform rollouts to get samples. In this tutorial,
we will implement a custom worker to meet the requirements of the RL2 algorithm
:cite:`duan2016rl`. Specifically, RL2 samples multiple episodes in one trial and
resets the RNN policy state at the beginning of a trail.
```

## Worker interface and DefaultWorker

In Garage, all `Worker`s need to implement the [`Worker` interface](https://garage.readthedocs.io/en/latest/_autoapi/garage/sampler/index.html#garage.sampler.Worker).
Garage provides a [`DefaultWorker`](https://garage.readthedocs.io/en/latest/_autoapi/garage/sampler/index.html#garage.sampler.DefaultWorker)
for general algorithms. Below is the `rollout()` method of `DefaultWorker`, the
most important method of `Worker`.

```py
    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()
```

To implement a custom worker, an easy way is to override the methods we want to
change (e.g. `rollout()`) instead of implementing our custom worker from
scratch. For example:

```py
from garage.sampler import DefaultWorker

class MyWorker(DefaultWorker):
    """Initialize a worker.

    Args:
        seed (int): The seed to use to initialize random number generators.
        max_episode_length (int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number):
        ...
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)
        ...

    def rollout(self):
        ...
```

## Custom Worker for RL2

In RL2, different from general RL algorithms, multiple episodes are sampled
in one trial. Thus we can move the original rollout into a loop:

```py
from garage.sampler import DefaultWorker

class RL2Worker(DefaultWorker):
    """Initialize a worker for RL2.

    Args:
        seed (int): The seed to use to initialize random number generators.
        max_episode_length (int or float): The maximum length episode which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_episodes_per_trial (int): Number of episodes sampled per trial/
            meta batch. Policy resets in the beginning of a meta batch,
            and obtain `n_episodes_per_trial` episodes in one meta batch.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number,
            n_episodes_per_trial=2):
        self._n_episodes_per_trial = n_episodes_per_trial
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def rollout(self):
        for _ in range(self._n_episodes_per_trial):
            self.start_episode()
            while not self.step_episode():
                pass
        return self.collect_episode()
```

This is not enough, because RL2 only resets the RNN policy state at the
beginning of a trail. So we need to override `start_episode()`. And we want to
record the batch index.

```py
...
    def start_episode(self):
        self._episode_length = 0
        self._prev_obs = self.env.reset()

    def rollout(self):
        self.agent.reset()
        for _ in range(self._n_episodes_per_trial):
            self.start_episode()
            while not self.step_episode():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._rewards),
                                                 self._worker_number)
        return self.collect_episode()
```

We have completed our custom worker for RL2. For reference, you can see the
complete implementation of RL2 algorithm and its worker [here](https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2.py).

To use the custom worker in a launcher, just set the `worker_class` of the
trainer, for example:

```py
from garage.tf.algos.rl2 import RL2Worker

    trainer.setup(...,
                 worker_class=RL2Worker,
                 ...)
```

You can see a full example launcher of PPO with RL2 [here](https://github.com/rlworkgroup/garage/blob/master/examples/tf/rl2_ppo_halfcheetah.py).

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----
*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

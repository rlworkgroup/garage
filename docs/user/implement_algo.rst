.. _implement_algo:

===========================
Implementing New Algorithms
===========================

In this section, we will describe how to implement an RL algorithm using garage.
Note that this section assumes some level of familiarity with reinforcement
learning. For a more gentle introduction to the field of reinforcement learning
as a whole, we recommend consulting `OpenAI's Spinning Up
<https://spinningup.openai.com/en/latest/user/introduction.html>`_.

We will start by introducing the core :code:`RLAlgorithm` API used in garage,
then show how to implement the classical REINFORCE [1]_ algorithm, also known as the
"vanilla" policy gradient (VPG).

Algorithm API
=============

All RL algorithms used with garage implement a small interface that allows
accessing important services such as snapshotting, "plotting" (visualization of
the current policy in the environment), and resume.

The interface requires a single method, :code:`train(runner)`, which takes a
:code:`garage.experiment.LocalRunner`. The interface is defined in
:code:`garage.np.algos.RLAlgorithm`, but inheriting from this class isn't necessary.

Some additional functionality (such as sampling and plotting) require
additional fields to exist.

.. code-block:: python

  class RLAlgorithm(abc.ABC):
      """Base class for all the algorithms.

      Note:
          If the field sampler_cls exists, it will be by LocalRunner.setup to
          initialize a sampler.

      """

      # pylint: disable=too-few-public-methods

      @abc.abstractmethod
      def train(self, runner):
          """Obtain samplers and start actual training for each epoch.

          Args:
              runner (LocalRunner): LocalRunner is passed to give algorithm
                  the access to runner.step_epochs(), which provides services
                  such as snapshotting and sampler control.

          """

In order to implement snapshotting and resume, instances of :code:`RLAlgorithm`
are also expected to support the Python standard library's `pickle interface <https://docs.python.org/3/library/pickle.html#pickling-class-instances>`_.
Garage primitives such as environments, policies, Q functions, and value
functions already implement this interface, so no work is typically required to
implement it.


Basic Setup
===========

Garage components are fairly weakly coupled, meaning that different parts can
be used independently. However, for this purpose of this tutorial we'll use the
parts together in the way that's generally recommended.

At the core of garage is the assumption that the algorithm runs a series of
"epochs", which are a unit of time small enough that most services, such as
logging, will only have new results once per epoch.

The current epoch is constrolled by the algorithm using
:code:`LocalRunner.step_epochs()`.

.. code-block:: python

    class MyAlgorithm:

        def train(self, runner):
            epoch_stepper = runner.step_epochs()
            print('It is epoch 0')
            next(epoch_stepper)
            print('It is epoch 1')
            next(epoch_stepper)
            print('It is epoch 2')

In practice, it's used in a loop like this:


.. code-block:: python

    class MyAlgorithm:

        def train(self, runner):
            for epoch in runner.step_epochs():
                print('It is epoch', epoch)


Each time the epoch is stepped, various "services" update. For example, logs
are synchronized, snapshotting (for later resuming) may occur, the plotter will
update, etc.

When an experiment is resumed, the epoch :code:`train` will be called again,
but the first epoch yielded by :code:`step_epochs` will be the one after the
snapshot.

In order to use the :code:`LocalRunner`, we'll need a set up log directory.
This can be done manually, but for this tutorial we'll use the
:code:`wrap_experiment` function to do that for us.

We'll also want an environment to test our algorithm with.

.. code-block:: python

    from garage import wrap_experiment
    from garage.envs import PointEnv
    from garage.experiment import LocalRunner

    @wrap_experiment(log_dir='my_algorithm_logs', use_existing_dir=True)
    def debug_my_algorithm(ctxt):
        runner = LocalRunner(ctxt)
        env = PointEnv()
        algo = MyAlgorithm()
        runner.setup(algo, env)
        runner.train(n_epochs=3)

    debug_my_algorithm()


With the above file and the :code:`MyAlgorithm` definition above, it should be
possible to run :code:`MyAlgorithm`, and get it to print an output like the
following:

.. code-block:: text

  2020-05-18 14:11:49 | [debug_my_algorithm] Logging to my_algorithm_logs
  2020-05-18 14:11:49 | [debug_my_algorithm] Obtaining samples...
  It is epoch 0
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #0 | Saving snapshot...
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #0 | Saved
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #0 | Time 0.00 s
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #0 | EpochTime 0.00 s
  -------------  -
  TotalEnvSteps  0
  -------------  -
  It is epoch 1
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #1 | Saving snapshot...
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #1 | Saved
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #1 | Time 0.01 s
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #1 | EpochTime 0.00 s
  -------------  -
  TotalEnvSteps  0
  -------------  -
  It is epoch 2
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #2 | Saving snapshot...
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #2 | Saved
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #2 | Time 0.01 s
  2020-05-18 14:11:49 | [debug_my_algorithm] epoch #2 | EpochTime 0.00 s
  -------------  -
  TotalEnvSteps  0
  -------------  -

Now that we have the basics out of the way, we can start actually doing some
reinforcement learning.


Gathering Samples
=================

In the above section, we set up an algorithm, but never actually explored the
environment at all, as can be seen by :code:`TotalEnvSteps` always being zero.

In order to collect samples from the environment, we can set the
:code:`sampler_cls` and :code:`policy` fields on our algorithm, and call
:code:`runner.obtain_samples()`. We'll also need to seed the random number
generators used for the experiment.

.. code-block:: python

    from garage.samplers import RaySampler

    class MyAlgorithm:

        sampler_cls = RaySampler

        def __init__(self, env_spec, policy):
            self.env_spec = env_spec
            self.policy = policy
            self.max_path_length = 200

        def train(self, runner):
            for epoch in runner.step_epochs():
                samples = runner.obtain_samples(epoch)

    from garage import wrap_experiment
    from garage.envs import PointEnv
    from garage.experiment import LocalRunner
    from garage.experiment.deterministic import set_seed
    from garage.torch.policies import GaussianMLPPolicy

    @wrap_experiment(log_dir='my_algorithm_logs', use_existing_dir=True)
    def debug_my_algorithm(ctxt):
        set_seed(100)
        runner = LocalRunner(ctxt)
        env = PointEnv()
        policy = GaussianMLPPolicy(env.spec)
        algo = MyAlgorithm(policy)
        runner.setup(algo, env)
        runner.train(n_epochs=500, batch_size=4000)

    debug_my_algorithm()


Of course, we'll need to actually use the resulting samples to train our
policy. For the purposes of this tutorial, we'll implement an extremely simple
form of REINFORCE [1]_ (a.k.a. Vanilla Policy Gradient) using PyTorch.


.. code-block:: python

    import torch
    import numpy as np

    from garage.samplers import RaySampler
    from garage.misc import tensor_utils as tu

    class MyAlgorithm:

        sampler_cls = RaySampler

        def __init__(self, env_spec, policy):
            self.env_spec = env_spec
            self.policy = policy
            self.max_path_length = 100
            self._policy_opt = torch.optim.Adam(self.policy.parameters())
            self._discount = 0.99

        def train(self, runner):
            for epoch in runner.step_epochs():
                samples = runner.obtain_samples(epoch)
                self.train_once(samples)

        def train_once(self, samples):
            losses = []
            self._policy_opt.zero_grad()
            for path in samples:
                returns_numpy = tu.discount_cumsum(path['rewards'], self._discount)
                returns = torch.Tensor(returns_numpy.copy())
                obs = torch.Tensor(path['observations'])
                actions = torch.Tensor(path['actions'])
                log_likelihoods = self.policy.log_likelihood(obs, actions)
                loss = (-log_likelihoods * returns).mean()
                loss.backward()
                losses.append(loss.item())
            self._policy_opt.step()
            return np.mean(losses)

That lets us train a policy, but it doesn't let us confirm that it actually works.
We can add a little logging to the :code:`train()` method.


.. code-block:: python

    from dowel import tabular

    from garage import log_performance, TrajectoryBatch

    ...

        def train(self, runner):
            for epoch in runner.step_epochs():
                samples = runner.obtain_samples(epoch)
                log_performance(epoch,
                                TrajectoryBatch.from_trajectory_list(self.env_spec,
                                                                    samples),
                                self._discount)
                mean_loss = self.train_once(samples)
                tabular.record('VPGLoss', mean_loss)

Running the experiment file should now show us that it solves the PointEnv
after about 100 epochs.
For completeness, the full experiment file is repeated below:

.. code-block:: python

    import torch
    import numpy as np

    from dowel import tabular

    from garage import log_performance, TrajectoryBatch
    from garage.sampler import RaySampler
    from garage.misc import tensor_utils as tu

    class MyAlgorithm:

        sampler_cls = RaySampler

        def __init__(self, env_spec, policy):
            self.env_spec = env_spec
            self.policy = policy
            self.max_path_length = 200
            self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
            self._discount = 0.99

        def train(self, runner):
            for epoch in runner.step_epochs():
                samples = runner.obtain_samples(epoch)
                log_performance(epoch,
                                TrajectoryBatch.from_trajectory_list(self.env_spec,
                                                                    samples),
                                self._discount)
                mean_loss = self.train_once(samples)
                tabular.record('VPGLoss', mean_loss)

        def train_once(self, samples):
            losses = []
            self._policy_opt.zero_grad()
            for path in samples:
                returns_numpy = tu.discount_cumsum(path['rewards'], self._discount)
                returns = torch.Tensor(returns_numpy.copy())
                obs = torch.Tensor(path['observations'])
                actions = torch.Tensor(path['actions'])
                log_likelihoods = self.policy.log_likelihood(obs, actions)
                loss = (-log_likelihoods * returns).mean()
                loss.backward()
                losses.append(loss.item())
            self._policy_opt.step()
            return np.mean(losses)


    from garage import wrap_experiment
    from garage.envs import PointEnv, GarageEnv
    from garage.experiment import LocalRunner
    from garage.experiment.deterministic import set_seed
    from garage.torch.policies import GaussianMLPPolicy

    @wrap_experiment(log_dir='my_algorithm_logs', use_existing_dir=True, archive_launch_repo=False)
    def debug_my_algorithm(ctxt):
        set_seed(100)
        runner = LocalRunner(ctxt)
        env = GarageEnv(PointEnv())
        policy = GaussianMLPPolicy(env.spec)
        algo = MyAlgorithm(env.spec, policy)
        runner.setup(algo, env)
        runner.train(n_epochs=500, batch_size=4000)

    debug_my_algorithm()


.. [1] Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

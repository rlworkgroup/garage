.. _implement_mdp:

=============================
Implementing New Environments
=============================

Garage uses an environment API based on the very popular OpenAI Gym interface. The main difference is that garage uses :code:`akro` to describe input and output spaces, which are an extension of the :code:`gym.Space` API.

If you have an OpenAI Gym compatible environment, you can wrap it in :code:`garage.envs.GarageEnv` to use it with garage.

Garage also has a wrapper which supports Deepmind's :code:`dm_control` API, :code:`garage.envs.dm_control.DmControlEnv`.


In the rest of this section, we will walk through an example of implementing a
point robot environment using our framework. A more complete version of this
environment is available as :code:`garage.envs.PointEnv`.

We will implement a simple environment with 2D observations and 2D actions. The goal is
to control a point robot in 2D to move it to the origin. We receive position of
a point robot in the 2D plane :math:`(x, y) \in \mathbb{R}^2`. The action is
its velocity :math:`(\dot x, \dot y) \in \mathbb{R}^2` constrained so that
:math:`|\dot x| \leq 0.1` and :math:`|\dot y| \leq 0.1`. We encourage the robot
to move to the origin by defining its reward as the negative distance to the
origin: :math:`r(x, y) = - \sqrt{x^2 + y^2}`.

We start by creating a new file for the environment, then we declare a class inheriting from
the base environment and add some imports:

.. code-block:: python

    import akro
    import gym
    import numpy as np


    class PointEnv(gym.Env):

        # ...

For each environment, we will need to specify the set of valid observations and the
set of valid actions. This is done by implementing the following
property methods:

.. code-block:: python

    class PointEnv(Env):

        # ...

        @property
        def observation_space(self):
            return akro.Box(low=-np.inf, high=np.inf, shape=(2,))

        @property
        def action_space(self):
            return akro.Box(low=-0.1, high=0.1, shape=(2,))

The :code:`Box` space means that the observations and actions are 2D vectors
with continuous values. The observations can have arbitrary values, while the
actions should have magnitude at most 0.1.

Now onto the interesting part, where we actually implement the dynamics for the
MDP. This is done through two methods, :code:`reset` and
:code:`step`. The :code:`reset` method randomly initializes the state
of the environment according to some initial state distribution. To keep things
simple, we will just sample the coordinates from a uniform distribution. The
method should also return the initial observation. In our case, it will be the
same as its state.

.. code-block:: python

    class PointEnv(gym.Env):

        # ...

        def reset(self):
            self._state = np.random.uniform(-1, 1, size=(2,))
            observation = np.copy(self._state)
            return observation

The :code:`step` method takes an action and advances the state of the
environment. It should return a :code:`Step` object (which is a wrapper around
:code:`namedtuple`), containing the observation for the next time step, the reward,
a flag indicating whether the episode is terminated after taking the step, and optional
extra keyword arguments (whose values should be vectors only) for diagnostic purposes.
The procedure that interfaces with the environment is responsible for calling
:code:`reset` after seeing that the episode is terminated.

.. code-block:: python

    class PointEnv(Env):

        # ...

        def step(self, action):
            self._state = self._state + action
            x, y = self._state
            reward = - (x**2 + y**2) ** 0.5
            done = abs(x) < 0.01 and abs(y) < 0.01
            next_observation = np.copy(self._state)
            return next_observation, reward, done, None

Finally, we can implement some plotting to visualize what the MDP is doing. For
simplicity, let's just print the current state of the MDP on the terminal:

.. code-block:: python

    class PointEnv(gym.Env):

        # ...

        def render(self):
            print 'current state:', self._state

And we're done! We can now simulate the environment using the following diagnostic
script:

.. code-block:: bash

    python scripts/sim_env.py garage.envs.point_env --mode random

It simulates an episode of the environment with random actions, sampled from a
uniform distribution within the defined action bounds.

You could also train a neural network policy to solve the task, which is probably
overkill. To do so, create a new script with the following content (we will use
stub mode):


.. code-block:: python

    from garage import wrap_experiment
    from garage.envs import PointEnv
    from garage.envs import normalize
    from garage.experiment import LocalTFRunner
    from garage.experiment.deterministic import set_seed
    from garage.np.baselines import LinearFeatureBaseline
    from garage.tf.algos import TRPO
    from garage.tf.policies import CategoricalMLPPolicy


    @wrap_experiment
    def trpo_point(ctxt=None, seed=1):
        set_seed(seed)
        with LocalTFRunner(ctxt) as runner:
            env = normalize(PointEnv())

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        max_kl_step=0.01)

            runner.setup(algo, env)
            runner.train(n_epochs=100, batch_size=4000)


    trpo_point()

Assume that the file is :code:`examples/tf/trpo_point.py`. You can then run the script:

.. code-block:: bash

    python examples/tf/trpo_point.py

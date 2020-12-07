.. _implement_mdp:

===========================
Adding a New Environment
===========================

Garage uses an environment API based on the very popular OpenAI Gym interface. The main difference is that garage uses
:code:`akro` to describe input and output spaces, which are an extension of the :code:`gym.Space` API.

If you have an OpenAI Gym compatible environment, you can wrap it in :code:`garage.envs.GymEnv` to use it with
garage. You can also provide its environment name id (see the example below), and let :code:`GymEnv` creates
the environment for you.

.. code-block:: python

    import gym
    from garage.envs import GymEnv

    my_env = GymEnv('CartPole-v1')  # shorthand for GymEnv(gym.make('CartPole-v1'))

+++++++++++++++++++++++++++
Add an Environment Wrapper
+++++++++++++++++++++++++++

If you would like to add an existing environment to garage, you will probably need an environment wrapper to handle
environment-specific logic.

Available environment wrappers in garage are:

* :code:`garage.envs.dm_control.DmControlEnv`, which supports Deepmind's :code:`dm_control` API.
* :code:`garage.envs.bullet.BulletEnv`, which supports the :code:`pybullet` API.

To-do list for adding a new environment wrapper:

* **Specify** :code:`observation_space` and :code:`action_space`
These state the set of valid observations and the set of valid actions. Find a detailed example in the next section.

* **Implement** :code:`reset` and :code:`step`
Allow the environment to reset and step. Find a detailed example in the next section.

* (Optional) **Implement** :code:`__getstate__` and :code:`__setstate__`
Garage pickles the environment to save snapshots, whereas some attributes of your environment might not be pickle-able
(e.g. a client-server connection). If needed, provide your own implementation to make sure your environment is
pickle-able.

* (Optional) **Unit Tests**
If you would like to contribute your own environment back to garage, make sure to add unit
tests under the directory :code:`tests/garage/envs`. You can find examples of environment tests here as well.


+++++++++++++++++++++++++++
Implement a New Environment
+++++++++++++++++++++++++++

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


    class PointEnv(Environment):

        # ...

For each environment, we will need to specify the set of valid observations and the
set of valid actions. This is done by implementing the following
property methods:

.. testcode::

    class PointEnv(Environment):

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

.. testcode::

    class PointEnv(Environment):

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

.. testcode::

    class PointEnv(Environment):

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

.. testcode::

    class PointEnv(Environment):

        # ...

        def render(self):
            print ('current state:', self._state)

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
    from garage.experiment.deterministic import set_seed
    from garage.np.baselines import LinearFeatureBaseline
    from garage.sampler import LocalSampler
    from garage.tf.algos import TRPO
    from garage.tf.policies import CategoricalMLPPolicy
    from garage.trainer import TFTrainer


    @wrap_experiment
    def trpo_point(ctxt=None, seed=1):
        set_seed(seed)
        with TFTrainer(ctxt) as trainer:
            env = normalize(PointEnv())

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=0.99,
                        max_kl_step=0.01)

            trainer.setup(algo, env)
            trainer.train(n_epochs=100, batch_size=4000)


    trpo_point()

Assume that the file is :code:`examples/tf/trpo_point.py`. You can then run the script:

.. code-block:: bash

    python examples/tf/trpo_point.py


----

This page was authored by K.R. Zentner (`@krzentner <https://github.com/krzentner>`_), with contributions from Ryan Julian (`@ryanjulian <https://github.com/ryanjulian>`_), Jonathon Shen (`@jonashen <https://github.com/jonashen>`_), Rocky Duan (`@dementrock <https://github.com/dementrock>`_), Eric Yihan Chen (`@AiRuiChen <https://github.com/AiRuiChen>`_).

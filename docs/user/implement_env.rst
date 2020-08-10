.. _implement_mdp:

================================
Working with Garage Environments
================================

Garage uses the :code:`Environment` API for all RL environments.

The public API methods of :code:`Environment` are:

* :code:`reset()`: Resets the environment. Return the first observation, and the episode-level information.
* :code:`step()`: Steps the environment with an action and returns an EnvStep.

    An :code:`EnvStep` is a named tuple of :code:`env_spec`, :code:`observation`, :code:`action`, :code:`reward`,
    :code:`step_type` representing a single step in the environment.
* :code:`render()`: Returns value for rendering, could be a terminal text, an image, etc.
* :code:`visualize()`: Creates a visualization of the environment.
* :code:`close()`: Closes the environment.

The attributes of :code:`Environment` are:

* :code:`action_space`: The action space specification.
* :code:`observation_space`: The observation space specification.
* :code:`spec`: The environment specifications.
* :code:`render_modes`: The list of supported render modes.

You should expect to use all environments in garage with this API.

+++++++++++++++++++++++++++
Add an Environment Wrapper
+++++++++++++++++++++++++++

Luckily, Garage already came with a variety of external environment wrappers:

* :code:`garage.envs.GymEnv`, which supports the :code:`gym` API.
* :code:`garage.envs.dm_control.DmControlEnv`, which supports Deepmind's :code:`dm_control` API.
* :code:`garage.envs.bullet.BulletEnv`, which supports the :code:`pybullet` API.

    Note that :code:`pybullet` environments are :code:`gym`-based environments, and they can be wrapped with
    :code:`garage.envs.GymEnv` as well. In such case, :code:`garage.envs.GymEnv` will detect the bullet-based
    environment and returns a :code:`garage.envs.bullet.BulletEnv`.

If you would like to add an existing environment to garage, you will probably need an environment wrapper to handle
environment-specific logic.

Here is an handy To-do list for adding a new environment wrapper:

* **Inherit** the environment wrapper from :code:`Environment`.

* **Specify** attributes: :code:`observation_space`, :code:`action_space`, :code:`spec`, and :code:`render_modes`

    Note that :code:`observation_space` and :code:`action_space` need to be :code:`akro.Space`.

* **Implement** public methods: :code:`reset()`, :code:`step()`, :code:`render()`, :code:`visualize()`, :code:`close()`

* (Optional) **Implement** :code:`__getstate__` and :code:`__setstate__`

    Garage pickles the environment to save snapshots, whereas some attributes of your environment might not be
    pickle-able (e.g. a client-server connection). If needed, provide your own implementation to make sure your
    environment is pickle-able.

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
    import numpy as np

    from garage import Environment, EnvSpec, EnvStep, StepType


    class PointEnv(Environment):

        # ...

For each environment, we will need to specify the set of valid observations and the
set of valid actions, as well as the supported render modes. This is done by implementing
the following property methods:

.. testcode::

    class PointEnv(Environment):

        def __init__(self, max_episode_length=math.inf):
            self._step_cnt = 0
            self._max_episode_length = max_episode_length
            self._visualize = False

        @property
        def observation_space(self):
            return akro.Box(low=-np.inf, high=np.inf, shape=(2,))

        @property
        def action_space(self):
            return akro.Box(low=-0.1, high=0.1, shape=(2,))

         @property
        def spec(self):
            return EnvSpec(action_space=self.action_space,
                           observation_space=self.observation_space,
                           max_episode_length=self._max_episode_length)

        @property
        def render_modes(self):
            return [
                'ascii',
            ]

The :code:`Box` space means that the observations and actions are 2D vectors
with continuous values. The observations can have arbitrary values, while the
actions should have magnitude at most 0.1.

:code:`max_episode_length` indicates the maximum number of steps an episode can contain.
If the environment step count :code:`self._step_cnt` reaches this value, the environment
will produce a timeout signal indicated by :codeL`StepType.TIMEOUT` (found in :code:`EnvStep`
returned by :code:`step()`.)

Next, we can implement some plotting to visualize what the MDP is doing. For
simplicity, let's just print the current state of the MDP on the terminal:

.. testcode::

    class PointEnv(Environment):

        # ...

        def visualize(self):
            self._visualize = True

        def render(self, mode='ascii'):
            if mode == 'ascii':
                print ('current state:', self._state)

Now onto the interesting part, where we actually implement the dynamics for the
MDP. This is done through two methods, :code:`reset` and
:code:`step`. The :code:`reset` method randomly initializes the state
of the environment according to some initial state distribution. To keep things
simple, we will just sample the coordinates from a uniform distribution. The
method should also return the initial observation and the episode level information.
Note that the episode info dict could be needed to determine the first action
(e.g. in the case of goal-conditioned or MTRL.)
In our case, the initial observation will be the same as its state, and the episode
dict is empty as it is not needed for this simple environment.

.. testcode::

    class PointEnv(Environment):

        # ...

        def reset(self):
            self._state = np.random.uniform(-1, 1, size=(2,))
            observation = np.copy(self._state)

            return observation, dict()

The :code:`step` method takes an action and advances the state of the
environment. It should return a :code:`EnvStep` object (which is a wrapper around
:code:`namedtuple`), containing the environment specification, observation for the next time step,
the input action, the reward, an enum indicating the step type (could be :code:`FIRST`, :code:`MID`, :code:`TERMINAL`
or :code:`TIMEOUT`), and optional extra keyword arguments (whose values should be vectors only) for diagnostic purposes.

The procedure that interfaces with the environment is responsible for calling
:code:`reset` after seeing that the episode is terminated.

.. testcode::

    class PointEnv(Environment):

        # ...

        def step(self, action):
            self._state = self._state + action

            if self._visualize:  # print state to terminal if visualization mode is enabled
                self.render()

            x, y = self._state
            reward = - (x**2 + y**2) ** 0.5
            done = abs(x) < 0.01 and abs(y) < 0.01
            next_observation = np.copy(self._state)

            self._step_cnt += 1

            # determine the step type based on the done signal, step count, and max episode length
            if done:
                return StepType.TERMINAL
            elif step_cnt >= max_episode_length:
                return StepType.TIMEOUT
            elif step_cnt == 1:
                return StepType.FIRST
            else:
                return StepType.MID

            return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=next_observation,
                       env_info=dict(),
                       step_type=step_type)

And we're done! Lets create a simply rollout loop with this Point environment.

.. testcode::

    env = PointEnv(max_episode_length=10)
    policy = MyPolicy()
    first_observation, episode_info = env.reset()
    env.visualize()  # set visualization signal to True
    episode = []

    first_action = policy.get_action(first_observation)
    episode.append(env.step(first_action))
    while not episode[-1].step_type in (StepType.TERMINAL, StepType.TIMEOUT):
       action = policy.get_action(episode[-1].observation)
       episode.append(env.step(action))  # prints the current state to terminal
    env.close()


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
                        max_episode_length=100,
                        discount=0.99,
                        max_kl_step=0.01)

            runner.setup(algo, env)
            runner.train(n_epochs=100, batch_size=4000)


    trpo_point()

Assume that the file is :code:`examples/tf/trpo_point.py`. You can then run the script:

.. code-block:: bash

    python examples/tf/trpo_point.py


----

This page was authored by K.R. Zentner (`@krzentner <https://github.com/krzentner>`_), with contributions from Ryan Julian (`@ryanjulian <https://github.com/ryanjulian>`_), Jonathon Shen (`@jonashen <https://github.com/jonashen>`_), Rocky Duan (`@dementrock <https://github.com/dementrock>`_), Eric Yihan Chen (`@AiRuiChen <https://github.com/AiRuiChen>`_).

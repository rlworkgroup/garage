# Adding a New Environment

Garage uses the `Environment` API for all RL environments.

The public API methods of `Environment` are:

```eval_rst
+----------------------------------+--------------------------------------+
| Functions                        | Description                          |
+==================================+======================================+
| :meth:`~Environment.reset()`     | Resets the environment. Return the   |
|                                  | first observation, and the           |
|                                  | episode-level information.           |
+----------------------------------+--------------------------------------+
| :meth:`~Environment.step()`      | Steps the environment with an action |
|                                  | and returns an :class:`EnvSpec`.     |
|                                  | An `EnvStep` is a named tuple of     |
|                                  | `env_spec`, `observation`, `action`, |
|                                  | `reward`, `step_type` representing a |
|                                  | single step in the environment.      |
+----------------------------------+--------------------------------------+
| :meth:`~Environment.render()`    | Returns value for rendering, could   |
|                                  | be a terminal text, an image, etc.   |
+----------------------------------+--------------------------------------+
| :meth:`~Environment.visualize()` | Creates a visualization of the       |
|                                  | environment.                         |
+----------------------------------+--------------------------------------+
| :meth:`~Environment.close()`     | Closes the environment.              |
+----------------------------------+--------------------------------------+
```

The attributes of `Environment` are:
```eval_rst
+-----------------------+-------------------------------------------------+
| Properties            | Description                                     |
+=======================+=================================================+
| action_space          | The action space specification                  |
+-----------------------+-------------------------------------------------+
| observation_space     | The observation space specification             |
+-----------------------+-------------------------------------------------+
| spec                  | The environment specifications                  |
+-----------------------+-------------------------------------------------+
| render_modes          | The list of supported render modes              |
+-----------------------+-------------------------------------------------+
```
You should expect to use all environments in garage with this API.

## Add Support for an Existing Environment Library

Luckily, Garage already came with a variety of external environment wrappers:

* `GymEnv`, which supports the `gym` API.
* `DmControlEnv`, which supports Deepmind's `dm_control` API.
* `BulletEnv`, which supports the `pybullet` API.

    Note that `pybullet` environments are `gym`-based environments, and they
    can be wrapped with `GymEnv` as well. In such case,
    `GymEnv` will detect the bullet-based environment and returns
    a `BulletEnv`.

Find more about these environment wrappers [here](environment_libraries).

If you would like to add an existing environment to garage, you will probably
need an environment wrapper to handle environment-specific logic.

Here is an handy To-do list for adding a new environment wrapper:

* (Recommended) **Inherit** the environment wrapper from `Environment`.

    Note that this is not required in Python because of duck typing. Your
    custom environment class will work as long as it implements
    interface methods of `Environment`.

* **Implement** attributes: `observation_space`, `action_space`, `spec`,
and `render_modes`

    Note that `observation_space` and `action_space` need to be `akro.Space`.

* **Implement** public methods: `reset()`, `step()`, `render()`, `visualize()`,
`close()`

* (Optional) **Implement** `__getstate__` and `__setstate__`

    Garage pickles the environment to save snapshots, whereas some attributes
    of your environment might not be pickle-able (e.g. a client-server
    connection). If needed, provide your own implementation to make sure your
    environment is pickle-able.

* (Optional) **Unit Tests**

    If you would like to contribute your own environment back to garage,
    make sure to add unit tests under the directory `tests/garage/envs`.
    You can find examples of environment tests here as well.


## Implement a New Environment

In the rest of this section, we will walk through an example of implementing a
point robot environment using our framework. A more complete version of this
environment is available as `PointEnv`.

```eval_rst
We will implement a simple environment with 2D observations and 2D actions. The
goal is to control a point robot in 2D to move it to the origin. We receive
position of a point robot in the 2D plane :math:`(x, y) \in \mathbb{R}^2`.
The action is its velocity :math:`(\dot x, \dot y) \in \mathbb{R}^2`
constrained so that :math:`|\dot x| \leq 0.1` and :math:`|\dot y| \leq 0.1`.
We encourage the robot to move to the origin by defining its reward as the
negative distance to the origin: :math:`r(x, y) = - \sqrt{x^2 + y^2}`.
```

We start by creating a new file for the environment, then we declare a class
inheriting from the base environment and add some imports:

```py

    import akro
    import numpy as np

    from garage import Environment, EnvSpec, EnvStep, StepType


    class PointEnv(Environment):

        # ...
```

For each environment, we will need to specify the set of valid observations
and the set of valid actions, as well as the supported render modes.
This is done by implementing the following property methods:

```py

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

```

The `Box` space means that the observations and actions are 2D vectors
with continuous values. The observations can have arbitrary values, while the
actions should have magnitude at most 0.1.

`max_episode_length` indicates the maximum number of steps an episode can contain.
If the environment step count `self._step_cnt` reaches this value, the environment
will produce a timeout signal indicated by :codeL`StepType.TIMEOUT` (found
in `EnvStep` returned by `step()`.)

Next, we can implement some plotting to visualize what the MDP is doing. For
simplicity, let's just print the current state of the MDP on the terminal:

```py

    class PointEnv(Environment):

        # ...

        def visualize(self):
            self._visualize = True

        def render(self, mode='ascii'):
            if mode == 'ascii':
                print ('current state:', self._state)
```

Now onto the interesting part, where we actually implement the dynamics for the
MDP. This is done through two methods, `reset` and
`step`. The `reset` method randomly initializes the state
of the environment according to some initial state distribution. To keep things
simple, we will just sample the coordinates from a uniform distribution. The
method should also return the initial observation and the episode level
information.
Note that the episode info dict could be needed to determine the first action
(e.g. in the case of goal-conditioned or MTRL.)
In our case, the initial observation will be the same as its state, and the
episode dict is empty as it is not needed for this simple environment.

```py

    class PointEnv(Environment):

        # ...

        def reset(self):
            self._state = np.random.uniform(-1, 1, size=(2,))
            observation = np.copy(self._state)

            return observation, dict()
```

The `step` method takes an action and advances the state of the
environment. It should return a `EnvStep` object (which is a wrapper around
`namedtuple`), containing the environment specification, observation for the
next time step, the input action, the reward, an enum indicating the step type
(could be `FIRST`, `MID`, `TERMINAL` or `TIMEOUT`), and optional extra keyword
arguments (whose values should be vectors only) for diagnostic purposes.

The procedure that interfaces with the environment is responsible for calling
`reset` after seeing that the episode is terminated.

```py

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
```

And we're done! Lets create a simply rollout loop with this Point environment.

```py

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
```

You could also train a neural network policy to solve the task, which is
probably overkill. To do so, create a new script with the following content
(we will use stub mode):


```py

    from garage import wrap_experiment
    from garage.envs import PointEnv
    from garage.envs import normalize
    from garage.experiment import TFTrainer
    from garage.experiment.deterministic import set_seed
    from garage.np.baselines import LinearFeatureBaseline
    from garage.tf.algos import TRPO
    from garage.tf.policies import CategoricalMLPPolicy


    @wrap_experiment
    def trpo_point(ctxt=None, seed=1):
        set_seed(seed)
        with TFTrainer(ctxt) as trainer:
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

            trainer.setup(algo, env)
            trainer.train(n_epochs=100, batch_size=4000)


    trpo_point()
```

Assume that the file is `examples/tf/trpo_point.py`. You can then run
the script:

```

    python examples/tf/trpo_point.py
```

----

*This page was authored by
K.R. Zentner ([@krzentner](https://github.com/krzentner)), with contributions
from Ryan Julian ([@ryanjulian](https://github.com/ryanjulian)),
Jonathon Shen ([@jonashen](https://github.com/jonashen)),
Rocky Duan ([@dementrock](https://github.com/dementrock)),
Eric Yihan Chen ([@AiRuiChen](https://github.com/AiRuiChen)).*

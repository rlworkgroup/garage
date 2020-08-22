# Environment Libraries

Garage supports a variety of external environment libraries for different RL
training purposes. This section introduces the environment libraries
supported by garage and how to work with them.

## OpenAI `gym`
OpenAI's [gym](https://github.com/openai/gym) comes with the default garage
installation. To use a `gym` environment, you should wrap it with
`garage.envs.GymEnv`.

For example:
```
    import gym
    from garage.envs import GymEnv

    env = GymEnv(gym.make('CarRacing-v0')
```

The wrapper `GymEnv` is required because it brings `gym` environments
to garage's `Environment` format.

Find more about the `Environment` API [here](implement_env).

Note that `GymEnv` can also take a string
argument and create the `gym` environment for you. In fact, this is the
**preferred** way to create a `gym` environment in garage.

For example:
```
    from garage.envs import GymEnv

    env = GymEnv('CarRacing-v0')
```

## DeepMind `dm_control`
DeepMind's [dm_control](https://github.com/deepmind/dm_control) can be
installed via `pip install 'garage[mujoco,dm_control]'`. Checkout the
[installation guide](installation) for details about setting up mujoco
dependencies for `dm_control`.

`dm_control` environments are wrapped by `garage.envs.dm_control.DMControlEnv`:

```
    from garage.envs.dm_control import DMControlEnv

    env = DMControlEnv.from_suite(domain_name, task_name)
```

Note that `DMControlEnv.from_suite()` is a convenient (and preferred) function
that returns a `DMControlEnv` wrapping a `dm_control` environment with the
given domain and task name.

## MetaWorld
[MetaWorld](https://github.com/rlworkgroup/metaworld) provides environments
for benchmark for meta- and multi-task reinforcement learning. Since MetaWorld
environments implement the `gym` environment interface, they can be wrapped by
`garage.envs.GymEnv` as well.

```
    from metaworld.benchmarks import MT10
    from garage.envs import GymEnv

    task = MT10.get_train_tasks().all_task_names[0]
    env = GymEnv(MT10.from_task(task)
```

## PyBullet
[PyBullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)
provides environments supported by the [Bullet Physics SDK](https://github.com/bulletphysics/bullet3).
`PyBullet` dependencies can be installed via `pip install 'garage[bullet]'`.

`PyBullet` environments are wrapped by `garage.envs.bullet.BulletEnv`.

```
    from garage.envs.bullet import BulletEnv

    env = BulletEnv('KukaCamBulletEnv-v0')
```

Note that since `PyBullet` environments implement the `gym` environment
interface, they can be wrapped by `garage.envs.GymEnv` as well. In this case,
`GymEnv` will return a `BulletEnv` instance. For example:

```
    from garage.envs import GymEnv

    env = GymEnv('KukaCamBulletEnv-v0')
    # type(env) == BulletEnv
```

## More Environment Libraries?
Checkout [Adding a new Environment](implement_env) to find out  how to
create your own environment wrapper.

----

*This page was authored by Eric Yihan Chen
([@AiRuiChen](https://github.com/AiRuiChen)).*

# Environment

An environment in reinforcement learning is a task, or simulation, that an agent interacts with. Environments in garage are very similar to Open AI's [Gym](https://gym.openai.com/) environments. One of the main differences is that garage uses [akro](https://akro.readthedocs.io/en/latest/) to describe input and output spaces, which is an extension of the gym.Space API.


### How does a garage environment work?
RL occurs in an agent-environment loop, in which an agent performs an action in an environment, and the environment modifies its state and returns observations, from which an agent will decide on its next action, and so on.

In garage, this cycle is performed through two main functions: `reset()` -> `step()` (repeat).

The cycle begins by calling `reset()`, which sets the environment to an initial stage. This returns initial `observations` (containing initial observations) and `episode_info` (episode-level information), which an agent can use to determine its next action.

Actions determined by the agent and policy may then be passed into the `step()` function of the environment to update its state. This returns an `EnvStep`, which contains information about the step taken, such as the resulting reward and observation. `step()` is repeatedly called until an episode, or a trial, is determined to be over.

This is an example of the cycle described above, where an environment interacts with actions provided by a policy:
```python
env = MyEnv()
policy = MyPolicy()
first_observation, episode_info = env.reset()
env.visualize()  # visualization window opened

episode = []
# Determine the first action
first_action = policy.get_action(first_observation, episode_info)
episode.append(env.step(first_action))

while not episode[-1].last():
    action = policy.get_action(episode[-1].observation)
    episode.append(env.step(action))

env.close()  # visualization window closed

```

The `garage.envs` API reference outlines more functions and properties in detail, and can be found [here](<../_autoapi/garage/envs/index.html>).


### What environments does garage support?
Garage supports a variety of external environment libraries for different RL training purposes. These environments can be found [here](environment_libraries).

### How do I add an environment?
A tutorial on how to use environments can be found [here](implement_env).


----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

# Data Structures
garage uses several data structures for information transfer. They are used to interact with environments and algorithms, and other data structures. ***


### TimeStep
A `TimeStep` represents a single sample when an agent interacts with an environment. It describes a SARS (State–action–reward–state) tuple that characterizes the evolution of a MDP. Along with the data stored in its attributes, it comes with various properties and can be created from a EnvStep. 

See its API reference [here](<../_autoapi/garage/index.html#garage.TimeStep>). 



### TimeStepBatch
A `TimeStepBatch` is a data type used for off-policy algorithms, imitation learning and batch-RL. It is a tuple representing a batch of TimeSteps. Along with the data stored in its attributes, it comes with methods that concatenate and split TimeStepBatches, convert between TimeStepBatches and lists, and convert EpisodeBatch into TimeStepBatch. 

See its API reference [here](<../_autoapi/garage/index.html#garage.TimeStepBatch>). 



### EpisodeBatch
An `EpisodeBatch` is a data type used for on-policy algorithms. It represents a batch of whole episodes, produced when one or more agents interacts with one or more environments. Along with the data stored in its attributes, it comes with methods that concatenate, split, and create EpisodeBatches from a list of episodes.

An example that uses most of its methods:
```python
from garage import EpisodeBatch
batches = []
while True:
    batch = worker.rollout()
    batches.append(batch)
episodeBatch = EpisodeBatch.concatenate(*batches)
episodeBatches = episodeBatch.split()
listOfDicts = episodeBatch.to_list()
episodeBatchFromList = EpisodeBatch.from_list() #env_spec, paths



```

See its API reference [here](<../_autoapi/garage/index.html#garage.EpisodeBatch>). 




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

----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

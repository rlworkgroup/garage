# Data Structures
garage uses several data structures for information transfer. They are used to interact with environments and algorithms, and other data structures. ***


### TimeStep
A `TimeStep` represents a single sample when an agent interacts with an environment. It describes a SARS (State–action–reward–state) tuple that characterizes the evolution of a MDP. Along with the data stored in its attributes, it comes with various properties and can be created from a EnvStep. 

See its API reference [here](<../_autoapi/garage/index.html#garage.TimeStep>). 

<!-- Examples on how to use it -->

<!-- Referene to test / example files -->

### TimeStepBatch
A `TimeStepBatch` is a data type used for off-policy algorithms, imitation learning and batch-RL. It is a tuple representing a batch of TimeSteps. Along with the data stored in its attributes, it comes with methods that concatenate and split TimeStepBatches, convert between TimeStepBatches and lists, and convert EpisodeBatch into TimeStepBatch. 

<!-- Examples on how to use it -->

<!-- Referene to test / example files -->


See its API reference [here](<../_autoapi/garage/index.html#garage.TimeStepBatch>). 


### EpisodeBatch
An `EpisodeBatch` is a data type used for on-policy algorithms. It represents a batch of whole episodes, produced when one or more agents interacts with one or more environments. Along with the data stored in its attributes, it comes with methods that concatenate, split, and create EpisodeBatches from a list of episodes.

Below is an example that demonstrates several functions of EpisodeBatch:
```python
from garage import EpisodeBatch
batches = []
while True:
    batch = worker.rollout()
    batches.append(batch)
episodeBatch = EpisodeBatch.concatenate(*batches) # concatenates multiple batches into one
episodeBatches = episodeBatch.split() # splits EpisodeBatch into multiple
listOfDicts = episodeBatch.to_list() # converts EpisodeBatch into list of dicts
episodeBatchFromList = EpisodeBatch.from_list() # creates EpisodeBatch from a list
```

This is an example of EpisodeBatch used in the `train()` function in the VPG tutorial example:
```python
for epoch in trainer.step_epochs():
    samples = trainer.obtain_samples(epoch)
    log_performance(epoch,
                    EpisodeBatch.from_list(self.env_spec, samples),
                    self._discount)
    self._train_once(samples)
```


```
See its API reference [here](<../_autoapi/garage/index.html#garage.EpisodeBatch>). 

----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

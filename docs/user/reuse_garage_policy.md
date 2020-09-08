# Load and Use a Trained Policy

In this section you will learn how to extract a trained policy from an experiment
snapshot, as well as how to evaluate that policy in an environment.

## Obtaining an experiment snapshot

Please refer to [this page](save_load_resume_exp.md) for information on how to save
an experiment snapshot. The snapshot contains data such as:

- The trainer's `setup_args` and `train_args`
  - Random seed
  - Batch size
  - Number of epochs
  - And [more](https://github.com/rlworkgroup/garage/blob/175ac4c90a408e2314d91cdbe95e419b183e0684/src/garage/trainer.py#L393)
- The experiment's `stats`
- The environment
- The algorithm **(which includes the policy we want to evaluate)**

## Extracting a trained policy from a snapshot

To extract the trained policy from a saved experiment, you only need a few lines
of code:

```python
# Load the policy
from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session

snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load('path/to/snapshot/dir')
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']
```

This code makes use of a Garage `Snapshotter` instance. It calls
cloudpickle behind the scenes, and should continue to work even if we
change how we pickle (we used to use joblib, for example).

## Applying the policy to an environment

In order to use your newly-loaded trained policy, you first have to make sure that
the shapes of its observation and action spaces match those of the target environment.
An easy way of doing this is run the policy in the same environment in which it
was trained.

Once you have an environment initialized, the basic idea is this:

```python
steps, max_steps = 0, 150
done = False
obs = env.reset()  # The initial observation
policy.reset()

while steps < max_steps and not done:
    obs, rew, done, _ = env.step(policy.get_action(obs))
    env.render()  # Render the environment to see what's going on (optional)
    steps += 1

env.close()
```

This logic is bundled up in a more robust way in the `rollout()` function from
`garage.sampler.utils`. Let's bring everything together:

```python
# Load the policy and the env in which it was trained
from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session

snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load('path/to/snapshot/dir')
policy = data['algo'].policy
env = data['env']

# See what the trained policy can accomplish
from garage import rollout
path = rollout(env, policy, animated=True)
print(path)
```

----

_This page was authored by Hayden Shively
([@haydenshively](https://github.com/haydenshively))_

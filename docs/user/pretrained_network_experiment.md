# Use a pre-trained network to start a new experiment

In this tutorial, we will walk you through how to use a pre-trained network to start a new experiment in garage. In practice, garage allows users to save and load a pre-trained model to perform the task of interest in reinforcement learning.

In general starting a new experiment with a pre-trained network follow the same few steps:

- Initialize the pretrained model
- Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
- Define for the optimization algorithm which parameters we want to update during training
- Run the training step

```python
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
```

## Set up the environment

First, we will start by setting up the environment for the experiment. We will train a `PPO algorithm` as an example.

We will simply setup the environment by initializing the garage environment `GarageEnv` and the `LocalTFRunner` instance.

We use garage environment here in the example, but you can opt to use `gym` environment as well. The runner instance .....

```python

def ppo_pendulum(ctxt=None, seed=1):
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))

    ....
```

## Save and load a pre-trained model

After the environment setup, we can start to load a pre-trained model. We use [cloudpickle](https://github.com/cloudpipe/cloudpickle) (and/or [pickle](https://docs.python.org/3/library/pickle.html)) to load and save model object, i.e. `Policy`, `QFunction`.

We will discuss the following two mechanisms to save/ load a pre-trained model:

1. Directly work with pickle objects / files
2. Use garage snapshot

### Use garage snapshot

Garage has a `Snapshotter` class for creating and manupulating snapshot of the training models and data. To use a `Snapshotter` instance for an experiment, we simply need to pass a defined snapshot configuration as a parameter `snapshot_config` to a `LocalRunner` instance.

Define snapshot configuration as follows:

- snapshot_dir: directory of snapshot object
- snapshot_mode: type of snapshot
- snapshot_gap: gap number of snapshot

```python
ctxt=garage.experiment.SnapshotConfig(snapshot_dir=log_dir,
                                      snapshot_mode='last', # only last iteration will be saved
                                      snapshot_gap=1)
```

`ctxt` here will be pass to `LocalTFRunner` to initialize the snapshot object. If `None` is provided to `snapshot_config`, a default snapshot configuration will be used.

We will continue to define the model. For fine-tunning, we can create an algorithm with updated desired parameters, i.e. `optimizer_args`:

```python
        ...
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        # algo = PPO(
        #     env_spec=env.spec,
        #     policy=policy,
        #     baseline=baseline,
        #     max_path_length=100,
        #     discount=0.99,
        #     gae_lambda=0.95,
        #     lr_clip_range=0.2,
        #     optimizer_args=dict(
        #         batch_size=32,
        #         max_epochs=10,
        #     ),
        #     stop_entropy_gradient=True,
        #     entropy_method='max',
        #     policy_ent_coeff=0.02,
        #     center_adv=False,
        # )

        algo_updated = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                learning_rate=1e-2,  # updated parameter
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

```

Load the stored snapshot object from the directory. Unless specified, default snapshot directory should be at `data/local/experiment/`.

`restore()` will restore the original experiment from snapshot, meaning the original model objects and data will be used. Here, we want to restore the experiment and continue training with the algorithm using the parameters we want to update. 

```python

runner.restore(log_dir).setup(algo=algo_updated)

runner.resume()
```

### Directly work with pickle objects / files

To save a pre-trained model to an object, we simply use the `dumps` method to pickle the model object. The example shows how we can save the policy.

It is assumed that a saved model is existed in the local path.

```python

```


We will begin by loading a saved policy model.

```python
# Pytorch
import torch

PATH = './path_to/model' # model path

# specify the model
policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

# load the model
policy.load_state_dict(torch.load(PATH))

```

## Train

## Resources

### Pytorch pre-trained networks

https://pytorch.org/docs/stable/torchvision/models.html

# Use a Pre-Trained Network to Start a New Experiment

In this section you will learn how to load a pre-trained network and use it in
new experiments. In general, this process involves loading a snapshot, extracting
the component that you wish to reuse, and including that component in the new
experiment.

We'll cover two examples in particular:

- How to use a trained policy as an expert in Behavioral Cloning
- How to reuse a trained Q function in DQN

Before attempting either of these, you'll need a saved experiment snapshot. [This page](https://garage.readthedocs.io/en/latest/user/save_load_resume_exp.html)
will show you how to get one.

## Example: Use a pre-trained policy as a BC expert

There are two steps involved. First, we must load the pre-trained policy. Assuming
that it was trained with garage, details on extracting a policy from a saved experiment
can be found [here](https://garage.readthedocs.io/en/latest/user/reuse_garage_policy.html).
Next, we setup a new experiment and pass the policy as the `source` argument of
the `BC` constructor:

```python
# Load the policy
from garage.experiment import Snapshotter
snapshotter = Snapshotter()
snapshot = snapshotter.load('path/to/snapshot/dir')

expert = snapshot['algo'].policy
env = snapshot['env']  # We assume env is the same

# Setup new experiment
from garage import wrap_experiment
from garage.sampler import LocalSampler
from garage.torch.algos import BC
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer

@wrap_experiment
def bc_with_pretrained_expert(ctxt=None):
    trainer = Trainer(ctxt)
    policy = GaussianMLPPolicy(env.spec, [8, 8])
    batch_size = 1000
    sampler = LocalSampler(agents=expert,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              sampler=sampler,
              policy_lr=1e-2,
              loss='log_prob')
    trainer.setup(algo, env)
    trainer.train(100, batch_size=batch_size)


bc_with_pretrained_expert()
```

Please refer to [this page](https://garage.readthedocs.io/en/latest/user/algo_bc.html)
for more information on garage's implementation of Behavioral Cloning. If your expert
policy wasn't trained with garage, you can wrap it in garage's `Policy` API
(`garage.torch.policies.Policy`) before passing it to `BC`.

## Example: Use a pre-trained Q function in a new DQN experiment

Garage's DQN module accepts a Q function in its constructor: `DQN(env_space=env.spec, policy=policy, qf=qf, ...)`
To use a pre-trained Q function, we simply load one and pass it in, rather than
creating a new one. Since there is a relatively large number of constructs that
go into creating a DQN, we suggest you use the [Pong example code](https://github.com/rlworkgroup/garage/blob/master/examples/tf/dqn_pong.py)
as a starting point. You'll have to modify lines 68-75 (`qf = DiscreteCNNQFunction(...)`)
as shown below:

```python
import click
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.wrappers import ClipReward
from garage.envs.wrappers import EpisodicLife
from garage.envs.wrappers import FireReset
from garage.envs.wrappers import Grayscale
from garage.envs.wrappers import MaxAndSkip
from garage.envs.wrappers import Noop
from garage.envs.wrappers import Resize
from garage.envs.wrappers import StackFrames
from garage.experiment import Snapshotter  # Add this import!
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DQN
from garage.tf.policies import DiscreteQFArgmaxPolicy
from garage.trainer import TFTrainer

@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
@click.option('--max_episode_length', type=int, default=500)
@wrap_experiment
def dqn_pong(ctxt=None, seed=1, buffer_size=int(5e4), max_episode_length=500):
    """Train DQN on PongNoFrameskip-v4 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        buffer_size (int): Number of timesteps to store in replay buffer.
        max_episode_length (int): Maximum length of a path after which a path
            is considered complete. This is used during testing to minimize
            the memory required to store a single path.
    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        n_epochs = 100
        steps_per_epoch = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = gym.make('PongNoFrameskip-v4')
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = GymEnv(env, is_image=True)

        replay_buffer = PathBuffer(capacity_in_transitions=buffer_size)

        # MARK: begin modifications to existing example
        snapshotter = Snapshotter()
        snapshot = snapshotter.load('path/to/previous/run/snapshot/dir')
        qf = snapshot['algo']._qf
        # MARK: end modifications to existing example

        policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.02,
                                                 decay_ratio=0.1)

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_policy=exploration_policy,
                   replay_buffer=replay_buffer,
                   sampler=sampler,
                   qf_lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   double_q=False,
                   n_train_steps=500,
                   steps_per_epoch=steps_per_epoch,
                   target_network_update_freq=2,
                   buffer_batch_size=32)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_pong()
```

----

_This page was authored by Hayden Shively
([@haydenshively](https://github.com/haydenshively))_

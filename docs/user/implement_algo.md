# Implement a New Algorithm

In this section, we will describe how to implement an RL algorithm using garage.
Note that this section assumes some level of familiarity with reinforcement
learning. For a more gentle introduction to the field of reinforcement learning
as a whole, we recommend consulting [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html).

```eval_rst
We will start by introducing the core :code:`RLAlgorithm` API used in garage,
then show how to implement the classical REINFORCE :cite:`williams1992simple`
algorithm, also known as the "vanilla" policy gradient (VPG).
```

## Algorithm API

All RL algorithms used with garage implement a small interface that allows
accessing important services such as snapshotting, "plotting" (visualization of
the current policy in the environment), and resume.

The interface requires a single method, `train(trainer)`, which takes a
`garage.experiment.Trainer`. The interface is defined in
`garage.np.algos.RLAlgorithm`, but inheriting from this class isn't necessary.

Some additional functionality (such as sampling and plotting) require additional
fields to exist.

```eval_rst
.. literalinclude:: ../../src/garage/np/algos/rl_algorithm.py
```

In order to implement snapshotting and resume, instances of `RLAlgorithm`
are also expected to support the Python standard library's [pickle interface](https://docs.python.org/3/library/pickle.html#pickling-class-instances).
Garage primitives such as environments, policies, Q functions, and value
functions already implement this interface, so no work is typically required to
implement it.

## Basic Setup

Garage components are fairly weakly coupled, meaning that different parts can
be used independently. However, for this purpose of this tutorial we'll use the
parts together in the way that's generally recommended.

At the core of garage is the assumption that the algorithm runs a series of
"epochs", which are a unit of time small enough that most services, such as
logging, will only have new results once per epoch.

The current epoch is controlled by the algorithm using
`Trainer.step_epochs()`.

```py
class MyAlgorithm:

    def train(self, trainer):
        epoch_stepper = trainer.step_epochs()
        print('It is epoch 0')
        next(epoch_stepper)
        print('It is epoch 1')
        next(epoch_stepper)
        print('It is epoch 2')
```

In practice, it's used in a loop like this:

```py
class MyAlgorithm:

    def train(self, trainer):
        for epoch in trainer.step_epochs():
            print('It is epoch', epoch)
```

Each time the epoch is stepped, various "services" update. For example, logs
are synchronized, snapshotting (for later resuming) may occur, the plotter will
update, etc.

When an experiment is resumed, the epoch `train` will be called again, but the
first epoch yielded by `step_epochs` will be the one after the snapshot.

In order to use the `Trainer`, we'll need a set up log directory. This can
be done manually, but for this tutorial we'll use the `wrap_experiment` function
to do that for us.

We'll also want an environment to test our algorithm with.

```py
from garage import wrap_experiment
from garage.envs import PointEnv
from garage.experiment import Trainer

@wrap_experiment
def debug_my_algorithm(ctxt):
    trainer = Trainer(ctxt)
    env = PointEnv()
    algo = MyAlgorithm()
    trainer.setup(algo, env)
    trainer.train(n_epochs=3)

debug_my_algorithm()
```

With the above file and the `MyAlgorithm` definition above, it should be
possible to run `MyAlgorithm`, and get it to print an output like the following:

```sh
2020-07-22 23:32:34 | [debug_my_algorithm] Logging to /home/ruofu/garage/data/local/experiment/debug_my_algorithm
2020-07-22 23:32:34 | [debug_my_algorithm] Obtaining samples...
It is epoch 0
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #0 | Saving snapshot...
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #0 | Saved
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #0 | Time 0.01 s
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #0 | EpochTime 0.01 s
-------------  -
TotalEnvSteps  0
-------------  -
It is epoch 1
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #1 | Saving snapshot...
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #1 | Saved
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #1 | Time 0.01 s
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #1 | EpochTime 0.00 s
-------------  -
TotalEnvSteps  0
-------------  -
It is epoch 2
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #2 | Saving snapshot...
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #2 | Saved
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #2 | Time 0.02 s
2020-07-22 23:32:34 | [debug_my_algorithm] epoch #2 | EpochTime 0.01 s
-------------  -
TotalEnvSteps  0
-------------  -
```

Now that we have the basics out of the way, we can start actually doing some
reinforcement learning.

## Gathering Samples

In the above section, we set up an algorithm, but never actually explored the
environment at all, as can be seen by `TotalEnvSteps` always being zero.

In order to collect samples from the environment, we need to construct a
`sampler` and set it a field in our algorithm. Then we can call
`trainer.obtain_samples()` to get samples. We'll also need to seed the random
number generators used for the experiment.

```py
class SimpleVPG:

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = 200

    def train(self, trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)

from garage import wrap_experiment
from garage.envs import PointEnv
from garage.experiment import Trainer
from garage.experiment.deterministic import set_seed
from garage.samplers import LocalSampler
from garage.torch.policies import GaussianMLPPolicy

@wrap_experiment
def debug_my_algorithm(ctxt):
    set_seed(100)
    trainer = Trainer(ctxt)
    env = PointEnv()
    policy = GaussianMLPPolicy(env.spec)
    sampler = LocalSampler(agents=policy, envs=env, max_episode_length=200)
    algo = SimpleVPG(env.spec, policy, sampler)
    trainer.setup(algo, env)
    trainer.train(n_epochs=500, batch_size=4000)

debug_my_algorithm()
```

## Training the Policy with Samples

```eval_rst
Of course, we'll need to actually use the resulting samples to train our
policy with PyTorch, TensorFlow or NumPy. In this tutorial, we'll implement an
extremely simple form of REINFORCE :cite:`williams1992simple` (a.k.a. Vanilla
Policy Gradient) using PyTorch and TensorFlow. We will also implement a simple
Cross Entropy Method (CEM) :cite:`rubinstein2004cross` using NumPy.
```

### PyTorch

```py
import torch
import numpy as np

from garage.samplers import LocalSampler
from garage.np import discount_cumsum

class SimpleVPG:

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = 200
        self._discount = 0.99
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train(self, trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            self._train_once(samples)

    def _train_once(self, samples):
        losses = []
        self._policy_opt.zero_grad()
        for path in samples:
            returns_numpy = discount_cumsum(path['rewards'], self._discount)
            returns = torch.Tensor(returns_numpy.copy())
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions'])
            dist = self.policy(obs)[0]
            log_likelihoods = dist.log_prob(actions)
            loss = (-log_likelihoods * returns).mean()
            loss.backward()
            losses.append(loss.item())
        self._policy_opt.step()
        return np.mean(losses)
```

That lets us train a policy, but it doesn't let us confirm that it actually works.
We can add a little logging to the `train()` method.

```py
from garage import log_performance, EpisodeBatch

...
    def train(self, trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(
                epoch,
                EpisodeBatch.from_episode_list(self.env_spec, samples),
                self._discount)
            self._train_once(samples)
```

For completeness, the full experiment file ([`example/torch/tutorial_vpg.py`](https://github.com/rlworkgroup/garage/blob/master/examples/torch/tutorial_vpg.py))
is repeated below:

```eval_rst
.. literalinclude:: ../../examples/torch/tutorial_vpg.py
```

Running the experiment file should print outputs like the following. The policy
should solve the `PointEnv` after 100 epochs (i.e. the `Evaluation/SuccessRate`
reaches 1).

```sh
2020-07-24 15:30:32 | [tutorial_vpg] Logging to /home/ruofu/garage/data/local/experiment/tutorial_vpg
Sampling  [####################################]  100%
2020-07-24 15:30:36 | [tutorial_vpg] epoch #0 | Saving snapshot...
2020-07-24 15:30:36 | [tutorial_vpg] epoch #0 | Saved
2020-07-24 15:30:36 | [tutorial_vpg] epoch #0 | Time 3.65 s
2020-07-24 15:30:36 | [tutorial_vpg] epoch #0 | EpochTime 3.65 s
----------------------------------  -----------
Evaluation/AverageDiscountedReturn   -78.1057
Evaluation/AverageReturn            -180.404
Evaluation/Iteration                   0
Evaluation/MaxReturn                 -36.996
Evaluation/MinReturn                -625.757
Evaluation/NumEpisodes                26
Evaluation/StdReturn                 143.39
Evaluation/SuccessRate                 0.384615
Evaluation/TerminationRate             0.384615
TotalEnvSteps                       4085
----------------------------------  -----------
2020-07-24 15:30:37 | [tutorial_vpg] epoch #1 | Saving snapshot...
2020-07-24 15:30:37 | [tutorial_vpg] epoch #1 | Saved
2020-07-24 15:30:37 | [tutorial_vpg] epoch #1 | Time 4.21 s
2020-07-24 15:30:37 | [tutorial_vpg] epoch #1 | EpochTime 0.55 s
----------------------------------  -----------
Evaluation/AverageDiscountedReturn   -77.1423
Evaluation/AverageReturn            -186.052
Evaluation/Iteration                   1
Evaluation/MaxReturn                 -19.9412
Evaluation/MinReturn                -458.353
Evaluation/NumEpisodes                28
Evaluation/StdReturn                 134.528
Evaluation/SuccessRate                 0.428571
Evaluation/TerminationRate             0.428571
TotalEnvSteps                       8202
----------------------------------  -----------
...
```

As `PointEnv` currently not supports visualization, If you want to visualize the
policy when training, you can solve an Gym environment, for example
`LunarLanderContinuous-v2`, and set `plot` to `True` in `trainer.train()`:

```py
...
@wrap_experiment
def tutorial_vpg(ctxt=None):
    set_seed(100)
    trainer = Trainer(ctxt)
    env = GymEnv('LunarLanderContinuous-v2')
    policy = GaussianMLPPolicy(env.spec)
    sampler = LocalSampler(agents=policy, envs=env, max_episode_length=200)
    algo = SimpleVPG(env.spec, policy, sampler)
    trainer.setup(algo, env)
    trainer.train(n_epochs=500, batch_size=4000, plot=True)
...
```

### TensorFlow

Before the training part, TensorFlow version is almost the same as PyTorch's,
except for the replacement of `Trainer` with `TFTrainer`.

```py
...
from garage import wrap_experiment
from garage.envs import PointEnv
from garage.experiment import TFTrainer
from garage.experiment.deterministic import set_seed
from garage.tf.policies import GaussianMLPPolicy

@wrap_experiment
def tutorial_vpg(ctxt=None):
    set_seed(100)
    with TFTrainer(ctxt) as trainer:
        env = PointEnv()
        policy = GaussianMLPPolicy(env.spec)
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=200.
                               is_tf_worker=True)
        algo = SimpleVPG(env.spec, policy, sampler)
        trainer.setup(algo, env)
        trainer.train(n_epochs=500, batch_size=4000)
...
```

Different from PyTorch's version, we need to build the computation graph before
training the policy in TensorFlow.

```py
import tensorflow as tf
...

class SimpleVPG:

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = 200
        self._discount = 0.99
        self.init_opt()

    def init_opt(self):
        observation_dim = self.policy.observation_space.flat_dim
        action_dim = self.policy.action_space.flat_dim
        with tf.name_scope('inputs'):
            self._observation = tf.compat.v1.placeholder(
                tf.float32, shape=[None, observation_dim], name='observation')
            self._action = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, action_dim],
                                                    name='action')
            self._returns = tf.compat.v1.placeholder(tf.float32,
                                                     shape=[None],
                                                     name='return')
        policy_dist = self.policy.build(self._observation, name='policy').dist
        with tf.name_scope('loss'):
            ll = policy_dist.log_prob(self._action, name='log_likelihood')
            loss = -tf.reduce_mean(ll * self._returns)
        with tf.name_scope('train'):
            self._train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
                loss)
```

The `train()` method is the same, while int the `_train_once()` method, we feed
the inputs with sample data.

```py
    def train(self, trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(
                epoch,
                EpisodeBatch.from_list(self.env_spec, samples),
                self._discount)
            self._train_once(samples)

    def _train_once(self, samples):
        obs = np.concatenate([path['observations'] for path in samples])
        actions = np.concatenate([path['actions'] for path in samples])
        returns = []
        for path in samples:
            returns.append(discount_cumsum(path['rewards'], self._discount))
        returns = np.concatenate(returns)
        sess = tf.compat.v1.get_default_session()
        sess.run(self._train_op,
                 feed_dict={
                     self._observation: obs,
                     self._action: actions,
                     self._returns: returns,
                 })
        return np.mean(returns)
```

As it is mentioned above, to support snapshot and resume, we need to implement
all things pickling. However, we use instance variables (e.g. `self._action`)
to save unpickled `tf.Tensor` in the class. So we need to define `__getstate__`
and `__setstate__` like:

```py
   def __getstate__(self):
        data = self.__dict__.copy()
        del data['_observation']
        del data['_action']
        del data['_returns']
        del data['_train_op']
        return data

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_opt()
```

For completeness, the full experiment file ([`example/tf/tutorial_vpg.py`](https://github.com/rlworkgroup/garage/blob/master/examples/tf/tutorial_vpg.py))
is repeated below:

```eval_rst
.. literalinclude:: ../../examples/tf/tutorial_vpg.py
```

Similar to the PyTorch's version, Running the experiment file should print
outputs like the following. The policy should solve the `PointEnv` after 100
epochs (i.e. the `Evaluation/SuccessRate` reaches 1).

```sh
2020-07-24 15:31:44 | [tutorial_vpg] Logging to /home/ruofu/garage/data/local/experiment/tutorial_vpg_1
2020-07-24 15:31:45 | [tutorial_vpg] Obtaining samples...
Sampling  [####################################]  100%
2020-07-24 15:31:50 | [tutorial_vpg] epoch #0 | Saving snapshot...
2020-07-24 15:31:51 | [tutorial_vpg] epoch #0 | Saved
2020-07-24 15:31:51 | [tutorial_vpg] epoch #0 | Time 5.25 s
2020-07-24 15:31:51 | [tutorial_vpg] epoch #0 | EpochTime 5.25 s
----------------------------------  ----------
Evaluation/AverageDiscountedReturn   -376.475
Evaluation/AverageReturn            -1035.36
Evaluation/Iteration                    0
Evaluation/MaxReturn                 -969.42
Evaluation/MinReturn                -1090.39
Evaluation/NumEpisodes                 20
Evaluation/StdReturn                   35.3741
Evaluation/SuccessRate                  0
Evaluation/TerminationRate              0
TotalEnvSteps                        4000
----------------------------------  ----------
Sampling  [####################################]  100%
2020-07-24 15:31:53 | [tutorial_vpg] epoch #1 | Saving snapshot...
2020-07-24 15:31:53 | [tutorial_vpg] epoch #1 | Saved
2020-07-24 15:31:53 | [tutorial_vpg] epoch #1 | Time 7.42 s
2020-07-24 15:31:53 | [tutorial_vpg] epoch #1 | EpochTime 2.16 s
----------------------------------  ----------
Evaluation/AverageDiscountedReturn   -376.199
Evaluation/AverageReturn            -1044.4
Evaluation/Iteration                    1
Evaluation/MaxReturn                 -865.945
Evaluation/MinReturn                -1154.95
Evaluation/NumEpisodes                 20
Evaluation/StdReturn                   69.6729
Evaluation/SuccessRate                  0
Evaluation/TerminationRate              0
TotalEnvSteps                        8000
----------------------------------  ----------
...
```

### NumPy

We will implement [CEM](https://github.com/rlworkgroup/garage/blob/master/src/garage/np/algos/cem.py)
with NumPy, and train the `CategoricalMLPPolicy` to solve `CartPole-v1`. The
experiment function is similar to that of TensorFlow:

```py
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import TFTrainer
from garage.experiment.deterministic import set_seed
from garage.tf.policies import CategoricalMLPPolicy

@wrap_experiment
def tutorial_cem(ctxt=None):
    set_seed(100)
    with TFTrainer(ctxt) as trainer:
        env = GymEnv('CartPole-v1')
        policy = CategoricalMLPPolicy(env.spec)
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=200,
                               is_tf_worker=True)
        algo = SimpleCEM(env.spec, policy, sampler)
        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=1000)
```

When training the policy, we use `policy.get_param_values()` method to get the
initial parameters of the policy, and use `policy.set_param_values()` to update
parameters of the policy.

```py
import numpy as np
from garage.np import discount_cumsum
from garage.sampler import LocalSampler

class SimpleCEM:

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = 200
        self._discount = 0.99
        self._extra_std = 1
        self._extra_decay_time = 100
        self._n_samples = 20
        self._n_best = 1
        self._cur_std = 1
        self._cur_mean = self.policy.get_param_values()
        self._all_avg_returns = []
        self._all_params = [self._cur_mean.copy()]
        self._cur_params = None

    def train(self, trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(
                epoch,
                EpisodeBatch.from_list(self.env_spec, samples),
                self._discount)
            self._train_once(epoch, samples)

    def _train_once(self, epoch, paths):
        returns = []
        for path in paths:
            returns.append(discount_cumsum(path['rewards'], self._discount))
        avg_return = np.mean(np.concatenate(returns))
        self._all_avg_returns.append(avg_return)
        if (epoch + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_avg_returns)
            best_inds = np.argsort(-avg_rtns)[:self._n_best]
            best_params = np.array(self._all_params)[best_inds]
            self._cur_mean = best_params.mean(axis=0)
            self._cur_std = best_params.std(axis=0)
            self.policy.set_param_values(self._cur_mean)
            avg_return = max(self._all_avg_returns)
            self._all_avg_returns.clear()
            self._all_params.clear()
        self._cur_params = self._sample_params(epoch)
        self._all_params.append(self._cur_params.copy())
        self.policy.set_param_values(self._cur_params)
        return avg_return

    def _sample_params(self, epoch):
        extra_var_mult = max(1.0 - epoch / self._extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self._cur_std) +
            np.square(self._extra_std) * extra_var_mult)
        return np.random.standard_normal(len(
            self._cur_mean)) * sample_std + self._cur_mean
```

You can see the full experiment file [here](https://github.com/rlworkgroup/garage/blob/master/examples/np/tutorial_cem.py).
Running the experiment file should print outputs like the following. If you want
to visualize the policy when training, you can set `plot` to `True` in
`trainer.train()` as mentioned before in PyTorch section.

```sh
2020-07-24 15:33:49 | [tutorial_cem] Logging to /home/ruofu/garage/data/local/experiment/tutorial_cem
2020-07-24 15:33:50 | [tutorial_cem] Obtaining samples...
Sampling  [####################################]  100%
2020-07-24 15:33:54 | [tutorial_cem] epoch #0 | Saving snapshot...
2020-07-24 15:33:54 | [tutorial_cem] epoch #0 | Saved
2020-07-24 15:33:54 | [tutorial_cem] epoch #0 | Time 3.52 s
2020-07-24 15:33:54 | [tutorial_cem] epoch #0 | EpochTime 3.52 s
----------------------------------  ---------
Evaluation/AverageDiscountedReturn    20.0163
Evaluation/AverageReturn              22.5333
Evaluation/Iteration                   0
Evaluation/MaxReturn                  52
Evaluation/MinReturn                  10
Evaluation/NumEpisodes                45
Evaluation/StdReturn                   7.9822
Evaluation/TerminationRate             1
TotalEnvSteps                       1014
----------------------------------  ---------
2020-07-24 15:33:54 | [tutorial_cem] epoch #1 | Saving snapshot...
2020-07-24 15:33:54 | [tutorial_cem] epoch #1 | Saved
2020-07-24 15:33:54 | [tutorial_cem] epoch #1 | Time 4.03 s
2020-07-24 15:33:54 | [tutorial_cem] epoch #1 | EpochTime 0.50 s
----------------------------------  ----------
Evaluation/AverageDiscountedReturn    15.7595
Evaluation/AverageReturn              17.1017
Evaluation/Iteration                   1
Evaluation/MaxReturn                  24
Evaluation/MinReturn                  13
Evaluation/NumEpisodes                59
Evaluation/StdReturn                   2.75985
Evaluation/TerminationRate             1
TotalEnvSteps                       2023
----------------------------------  ----------
...
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by K.R. Zentner ([@krzentner](https://github.com/krzentner)) and Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

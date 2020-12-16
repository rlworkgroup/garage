# REINFORCE (VPG)

```eval_rst
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. :cite:`williams1992simple`                                                                                                           |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                 | .. figure:: ./images/tf.png                                                                                  |
|                   |    :scale: 10%                                                                                                   |    :scale: 20%                                                                                               |
|                   |    :class: no-scaled-link                                                                                        |    :class: no-scaled-link                                                                                    |
|                   |                                                                                                                  |                                                                                                              |
|                   |    PyTorch                                                                                                       |    TensorFlow                                                                                                |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.VPG <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.VPG>`_                     | `garage.tf.algos.VPG <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.VPG>`_                          |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/vpg.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/vpg.py>`_   | `garage/tf/algos/vpg.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/vpg.py>`_     |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Examples**      | `examples <algo_vpg.html#examples>`_                                                                                                                                                                                            |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

The REINFORCE algorithm, also sometimes known as Vanilla Policy Gradient (VPG), is the most basic policy gradient method, and was built upon to develop more complicated methods such as PPO and VPG.  The original paper on REINFORCE is available [here](https://link.springer.com/article/10.1007/BF00992696).

This doc will provide a high level overview of the algorithm and its implementation in garage. For a more thorough introduction into policy gradient methods as well reinforcement learning in general, we encourage you to read through Spinning Up [here](https://spinningup.openai.com/en/latest/).

Note that in the codebase, both the tensorflow and torch implementations refer to this algorithm as VPG.

## Overview

In REINFORCE as well as other policy gradient algorithms, the gradient steps taken aim to minimize a loss function by incrementally modifying the policy network's parameters. The loss function in the original REINFORCE algorithm is given by:

```math
-log(\pi(s,a)) * G
```

where the `log` term is the log probability of taking some action `a` at some state `s`, and `G`  is the return, the sum of the discounted rewards from the current timestep up until the end of the episode.

In practice, this loss function isn't typically used as its performance is limited by the high variance in the return `G` over entire episodes. To combat this, an advantage estimate is introduced in place of the return `G`. The advantage is given by:

```math
A(s,a) = r + \gamma V(s') - V(s)
```

Where `V(s)` is a learned value function that estimates the value of a given state, `r` is the reward received from transitioning from state `s` into state `s'` by taking action `a`, and `γ` is the discount rate, a hyperparameter passed to the algorithm.

The augmented loss function then becomes:

```math
-log(\pi(s,a)) * A(s,a)
```

Naturally, since the value function is learned over time as more updates are performed, it introduces some margin bias caused by the imperfect estimates, but decreases the overall variance as well.

In garage, a technique called Generalized Advantage Estimation is used to compute the advantage in the loss term. This introduces a hyperparameter λ that can be used to tune the amount of variance vs bias in each update, where λ=1 results in the maximum variance and zero bias, and λ=0 results in the opposite. Best results are typically achieved with `λ ϵ [0.9, 0.999]`. [This](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/) resource provides a more in-depth explanation of GAE and its utility.

## Examples

As with all algorithms in garage, you can take a look at the the examples provided in `garage/examples` for an idea of hyperparameter values and types . For VPG, these are:

### TF

```eval_rst
.. literalinclude:: ../../examples/tf/vpg_cartpole.py
```

### Pytorch

```eval_rst
.. literalinclude:: ../../examples/torch/vpg_pendulum.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

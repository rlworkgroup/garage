# Deep Deterministic Policy Gradient (DDPG)

```eval_rst
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Continuous control with deep reinforcement learning :cite:`lillicrap2015continuous`                                                                                                                                           |
+-------------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                 | .. figure:: ./images/tf.png                                                                                |
|                   |    :scale: 10%                                                                                                   |    :scale: 20%                                                                                             |
|                   |    :class: no-scaled-link                                                                                        |    :class: no-scaled-link                                                                                  |
|                   |                                                                                                                  |                                                                                                            |
|                   |    PyTorch                                                                                                       |    TensorFlow                                                                                              |
+-------------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.DDPG <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.DDPG>`_                   | `garage.tf.algos.DDPG <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.DDPG>`_                      |
+-------------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/ddpg.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/ddpg.py>`_ | `garage/tf/algos/ddpg.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/ddpg.py>`_ |
+-------------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
| **Examples**      | `torch/ddpg_pendulum <algo_ddpg.html#pytorch>`_                                                                  | `tf/ddpg_pendulum <algo_ddpg.html#tensorflow>`_                                                            |
+-------------------+------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
```

DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. It uses a supervised
method to update the critic network and policy gradient to update the actor
network. And there are exploration strategy, replay buffer and target networks
involved to stabilize the training process.

## Examples

Garage has implementations of DDPG with PyTorch and TensorFlow.

### PyTorch

```eval_rst
.. literalinclude:: ../../examples/torch/ddpg_pendulum.py
```

### TensorFlow

```eval_rst
.. literalinclude:: ../../examples/tf/ddpg_pendulum.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

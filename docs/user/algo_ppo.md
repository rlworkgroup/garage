# Proximal Policy Optimization

```eval_rst
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Proximal Policy Optimization Algorithms :cite:`schulman2017proximal`                                                                                                                                                      |
+-------------------+----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                               | .. figure:: ./images/tf.png                                                                              |
|                   |    :scale: 10%                                                                                                 |    :scale: 20%                                                                                           |
|                   |    :class: no-scaled-link                                                                                      |    :class: no-scaled-link                                                                                |
|                   |                                                                                                                |                                                                                                          |
|                   |    PyTorch                                                                                                     |    TensorFlow                                                                                            |
+-------------------+----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.PPO <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PPO>`_                   | `garage.tf.algos.PPO <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.PPO>`_                      |
+-------------------+----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/ppo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/ppo.py>`_ | `garage/tf/algos/ppo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/ppo.py>`_ |
+-------------------+----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| **Examples**      | `examples <algo_ppo.html#examples>`_                                                                                                                                                                                      |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Proximal Policy Optimization Algorithms (PPO) is a family of policy gradient
methods which alternate between sampling data through interaction with the
environment, and optimizing a "surrogate" objective function using stochastic
gradient ascent.

```eval_rst
Garage's implementation also supports adding entropy bonus to the objective.
Two types of entropy approaches could be used here. Maximum entropy approach
adds the dense entropy to the reward for each time step, while entropy
regularization adds the mean entropy to the surrogate objective. See
:cite:`levine2018reinforcement` for more details.
```

## Examples

Garage has implementations of PPO with PyTorch and TensorFlow.

### PyTorch

```eval_rst
.. literalinclude:: ../../examples/torch/ppo_pendulum.py
```

### TensorFlow

```eval_rst
.. literalinclude:: ../../examples/tf/ppo_pendulum.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

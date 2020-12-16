# Trust Region Policy Optimization (TRPO)

```eval_rst
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Trust Region Policy Optimization :cite:`schulman2015trust`                                                                                                                                                                      |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                 | .. figure:: ./images/tf.png                                                                                  |
|                   |    :scale: 10%                                                                                                   |    :scale: 20%                                                                                               |
|                   |    :class: no-scaled-link                                                                                        |    :class: no-scaled-link                                                                                    |
|                   |                                                                                                                  |                                                                                                              |
|                   |    PyTorch                                                                                                       |    TensorFlow                                                                                                |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.TRPO <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.TRPO>`_                   | `garage.tf.algos.TRPO <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.TRPO>`_                        |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/trpo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/trpo.py>`_ | `garage/tf/algos/trpo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/trpo.py>`_   |
+-------------------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Examples**      | `examples <algo_trpo.html#examples>`_                                                                                                                                                                                           |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Trust Region Policy Optimization, or TRPO, is a policy gradient algorithm that builds on REINFORCE/VPG to improve performance. It introduces a KL constraint that prevents incremental policy updates from deviating excessively from the current policy, and instead mandates that it remains within a specified trust region. The TRPO paper is available [here](https://arxiv.org/abs/1502.05477).  Also, please see [Spinning Up's write up](https://spinningup.openai.com/en/latest/algorithms/trpo.html) for a detailed description of the inner workings of the algorithm.

## Examples

### TF

```eval_rst
.. literalinclude:: ../../examples/tf/trpo_cartpole.py
.. literalinclude:: ../../examples/tf/trpo_cubecrash.py
.. literalinclude:: ../../examples/tf/trpo_cartpole_recurrent.py
```

### Pytorch

```eval_rst
.. literalinclude:: ../../examples/torch/trpo_pendulum.py
.. literalinclude:: ../../examples/torch/trpo_pendulum_ray_sampler.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

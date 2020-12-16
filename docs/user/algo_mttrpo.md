# Multi-Task Trust Region Policy Optimization (MT-TRPO)

```eval_rst
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Trust Region Policy Optimization :cite:`schulman2015trust`, Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning :cite:`yu2019metaworld` |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                                                                          |
|                   |    :scale: 10%                                                                                                                                                            |
|                   |    :class: no-scaled-link                                                                                                                                                 |
|                   |                                                                                                                                                                           |
|                   |    PyTorch                                                                                                                                                                |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.TRPO <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.TRPO>`_                                                                            |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/trpo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/trpo.py>`_                                                          |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Examples**      | :ref:`mttrpo_metaworld_mt1_push`, :ref:`mttrpo_metaworld_mt10`, :ref:`mttrpo_metaworld_mt50`                                                                              |
+-------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Multi-Task Trust Region Policy Optimization (MT-TRPO) is a multi-task RL method
that aims to learn TRPO algorithm to maximize the average discounted return
across multiple tasks. The algorithm is evaluated on the average performance
over training tasks.

## Examples

### mttrpo_metaworld_mt1_push

This example is to train TRPO on Multi-Task 1 (MT1) push environment, in which
we learn a policy to perform push tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mttrpo_metaworld_mt1_push.py
```

### mttrpo_metaworld_mt10

This example is to train TRPO on Multi-Task 10 (MT10) environment, in which we
learn a policy to perform 10 different manipulation tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mttrpo_metaworld_mt10.py
```

### mttrpo_metaworld_mt50

This example is to train TRPO on Multi-Task 50 (MT50) environment, in which we
learn a policy to perform 10 different manipulation tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mttrpo_metaworld_mt50.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

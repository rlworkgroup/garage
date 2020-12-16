# Multi-Task Proximal Policy Optimization (MT-PPO)

```eval_rst
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning :cite:`yu2019metaworld`, Proximal Policy Optimization Algorithms :cite:`schulman2017proximal` |
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                                                                                    |
|                   |    :scale: 10%                                                                                                                                                                      |
|                   |    :class: no-scaled-link                                                                                                                                                           |
|                   |                                                                                                                                                                                     |
|                   |    PyTorch                                                                                                                                                                          |
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.PPO <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PPO>`_                                                                                        |
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/ppo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/ppo.py>`_                                                                      |
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Examples**      | :ref:`mtppo_metaworld_mt1_push`, :ref:`mtppo_metaworld_mt10`, :ref:`mtppo_metaworld_mt50`                                                                                           |
+-------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Multi-Task PPO is a multi-task RL method that aims to learn PPO algorithm to maximize the average discounted return across multiple tasks. The algorithm is evaluated on the average performance over training tasks.

## Examples

### mtppo_metaworld_mt1_push

This example is to train PPO on Multi-Task 1 (MT1) push environment, in which we learn a policy to perform push tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_mt1_push.py
```

### mtppo_metaworld_mt10

This example is to train PPO on Multi-Task 10 (MT10) environment, in which we learn a policy to perform 10 different manipulation tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_mt10.py
```

### mtppo_metaworld_mt50

This example is to train PPO on Multi-Task 50 (MT50) environment, in which we learn a policy to perform 50 different manipulation tasks.

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_mt50.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*

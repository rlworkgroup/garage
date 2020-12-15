# Model-Agnostic Meta-Learning (MAML)

```eval_rst
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks :cite:`finn2017modelagnostic`                                                                            |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                                                                           |
|                   |    :scale: 10%                                                                                                                                                             |
|                   |    :class: no-scaled-link                                                                                                                                                  |
|                   |                                                                                                                                                                            |
|                   |    PyTorch                                                                                                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.MAML <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.maml>`_                                                                             |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/maml.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/maml.py>`_                                                           |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Examples**      | :ref:`maml_ppo_half_cheetah_dir`, :ref:`maml_trpo_half_cheetah_dir`, :ref:`maml_trpo_metaworld_ml1_push`, :ref:`maml_trpo_metaworld_ml10`. :ref:`maml_trpo_metaworld_ml45` |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

MAML is a meta-learning algorithm that trains the parameters of a policy such that they generalize well to unseen tasks. In essence, this technique produces models that are good few shot learners and easy to fine-tune.

## Default Parameters

```python
meta_batch_size=40,
inner_lr=0.1,
outer_lr=1e-3,
num_grad_updates=1,
meta_evaluator=None,
evaluate_every_n_epochs=1
```

## Examples

### maml_ppo_half_cheetah_dir

```eval_rst
.. figure:: ./images/pytorch.png
        :scale: 10%
.. literalinclude:: ../../examples/torch/maml_ppo_half_cheetah_dir.py
```

### maml_trpo_half_cheetah_dir

```eval_rst
.. figure:: ./images/pytorch.png
        :scale: 10%
.. literalinclude:: ../../examples/torch/maml_trpo_half_cheetah_dir.py
```

### maml_trpo_metaworld_ml1_push

```eval_rst
.. figure:: ./images/pytorch.png
        :scale: 10%
.. literalinclude:: ../../examples/torch/maml_trpo_metaworld_ml1_push.py
```

### maml_trpo_metaworld_ml10

```eval_rst
.. figure:: ./images/pytorch.png
        :scale: 10%
.. literalinclude:: ../../examples/torch/maml_trpo_metaworld_ml10.py
```

### maml_trpo_metaworld_ml45

```eval_rst
.. figure:: ./images/pytorch.png
        :scale: 10%
.. literalinclude:: ../../examples/torch/maml_trpo_metaworld_ml45.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

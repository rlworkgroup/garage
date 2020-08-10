# Behavioral Cloning

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Model-Free Imitation Learning with Policy Optimization :cite:`ho2016model`
   * - **Framework(s)**
     - .. figure:: ./images/pytorch.png
        :scale: 10%

        PyTorch
   * - **API Reference**
     - `garage.torch.algos.BC <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.BC>`_
   * - **Code**
     - `garage/torch/algos/bc.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/bc.py>`_
   * - **Examples**
     - :ref:`bc_point`, :ref:`bc_point_deterministic_policy`
```

Behavioral cloning is a simple immitation learning algorithm which maxmizes the likelhood of an expert demonstration's actions under the apprentice policy using direct policy optimization. Garage's implementation may use either a policy or dataset as the expert.

## Default Parameters

```python
policy_optimizer = torch.optim.Adam
policy_lr = 1e-3
loss = 'log_prob'
batch_size = 1000
```

## Examples

### bc_point

```eval_rst
.. literalinclude:: ../../examples/torch/bc_point.py
```

### bc_point_deterministic_policy

#### Experiment Results

![BC Mean Loss](images/bc_meanLoss.png) ![BC Mean Loss](images/bc_stdLoss.png)

```eval_rst
.. literalinclude:: ../../examples/torch/bc_point_deterministic_policy.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)) with contributions from Ryan Julian ([@ryanjulian](https://github.com/ryanjulian)).*

# Behavioral Cloning

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Model-Free Imitation Learning with Policy Optimization :cite:`ho2016model`
   * - **Framework(s)**
     - PyTorch
   * - **API Reference**
     - `garage.torch.algos.BC <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.BC>`_
   * - **Code**
     - `garage/torch/algos/bc.py <../_modules/garage/torch/algos/bc.html#BC>`_
   * - **Examples**
     - :ref:`bc_point`, :ref:`bc_point_deterministic_policy`
```

Behavioral cloning is a simple immitation learning algorithm which maxmizes the likelhood of an expert demonstration's actions under the apprentice policy using direct policy optimization. garage's implementation may use either a policy or dataset as the expert.


## Default Parameters

```python
policy_optimizer = torch.optim.Adam
policy_lr = 1e-3
loss = 'log_prob'
minibatches_per_epoch = 16
```

## Examples

### bc_point

```eval_rst
.. literalinclude:: ../../examples/torch/bc_point.py
```

### bc_point_deterministic_policy

```eval_rst
.. literalinclude:: ../../examples/torch/bc_point_deterministic_policy.py
```

## References
```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

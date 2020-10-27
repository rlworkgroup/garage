# Cross Entropy Method

```eval_rst
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | The cross-entropy method: A unified approach to Monte Carlo simulation, randomized optimization and machine learning :cite:`rubinstein2004cross` |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/numpy.png                                                                                                                   |
|                   |    :scale: 40%                                                                                                                                   |
|                   |    :class: no-scaled-link                                                                                                                        |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.np.algos.CEM <../_autoapi/garage/np/algos/index.html#garage.np.algos.CEM>`_                                                              |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/np/algos/cem.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/np/algos/cem.py>`_                                         |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
```

Cross Entropy Method (CEM) works by iteratively optimizing a gaussian
distribution of policy.

In each epoch, CEM does the following:

1. Sample n_samples policies from a gaussian distribution of mean cur_mean and
std cur_std.

2. Collect episodes for each policy.

3. Update cur_mean and cur_std by doing Maximum Likelihood Estimation over the
n_best top policies in terms of return.

## Examples

### NumPy

```eval_rst
.. literalinclude:: ../../examples/np/cem_cartpole.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

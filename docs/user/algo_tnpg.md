# Truncated Natural Policy Gradient (TNPG)

```eval_rst
+-------------------+--------------------------------------------------------------------------------------------------------------+
| **Paper**         | Benchmarking Deep Reinforcement Learning for Continuous Control :cite:`duan2016benchmarking`, A Natural      |
|                   | Policy Gradient :cite:`10.5555/2980539.2980738`                                                              |
+-------------------+--------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/tf.png                                                                                  |
|                   |    :scale: 10%                                                                                               |
|                   |    :class: no-scaled-link                                                                                    |
|                   |                                                                                                              |
|                   |    Tensorflow                                                                                                |
+-------------------+--------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.tf.algos.TNPG <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.TNPG>`_                        |
+-------------------+--------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/tf/algos/tnpg.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/tnpg.py>`_   |
+-------------------+--------------------------------------------------------------------------------------------------------------+
```

```eval_rst
Truncated Natural Policy Gradient develops upon the Natural Policy Gradient, which optimizes a policy for the maximum discounted rewards by gradient descent. TNPG a conjugate gradient algorithm to compute the natural policy gradient, cutting the computation cost when there are high-dimensional parameters. See :cite:`duan2016benchmarking` for more details.
```

## Default Parameters

```py
discount=0.99,
gae_lambda=0.98,
lr_clip_range=0.01,
max_kl_step=0.01,
policy_ent_coeff=0.0,
entropy_method='no_entropy',
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```
----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

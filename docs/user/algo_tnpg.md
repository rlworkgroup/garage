# Truncated Natural Policy Gradient

```eval_rst
+-------------------+--------------------------------------------------------------------------------------------------------------+
| **Paper**         | Model-Free Imitation Learning with Policy Optimization :cite:`ho2016model`                                   |
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
| **Examples**      |                                                                                                              |
+-------------------+--------------------------------------------------------------------------------------------------------------+
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

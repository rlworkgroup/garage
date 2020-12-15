# Episodic Reward Weighted Regression (ERWR)

```eval_rst
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Papers**        | Using Reward-weighted Regression for Reinforcement Learning of Task Space Control :cite:`peters2007reward`                                                                 |
|                   |                                                                                                                                                                            |
|                   | Policy Search for Motor Primitives in Robotics :cite:`2009koberpolicy`                                                                                                     |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/tf.png                                                                                                                                                |
|                   |    :scale: 20%                                                                                                                                                             |
|                   |    :class: no-scaled-link                                                                                                                                                  |
|                   |                                                                                                                                                                            |
|                   |    Tensorflow                                                                                                                                                              |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.tf.algos.ERWR <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.ERWR>`_                                                                                      |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/tf/algos/erwr.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/erwr.py>`_                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Examples**      | :ref:`erwr_cartpole`                                                                                                                                                       |
+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Episodic Reward Weighted Regression (ERWR) is an extension of the original RWR algorithm, which uses a linear policy to solve the immediate rewards learning problem. The extension implemented here applies RWR to episodic reinforcement learning. To read more about both algorithms see the cited papers or the summary provided in [this](https://spiral.imperial.ac.uk:8443/bitstream/10044/1/12051/7/fnt_corrected_2014-8-22.pdf) text.

## Default Parameters

```python
scope=None
discount=0.99
gae_lambda=1
center_adv=True
positive_adv=True
fixed_horizon=False
lr_clip_range=0.01
max_kl_step=0.01
optimizer=None
optimizer_args=None
policy_ent_coeff=0.0
use_softplus_entropy=False
use_neg_logli_entropy=False
stop_entropy_gradient=False
entropy_method='no_entropy'
name='ERWR'
```

## Examples

### erwr_cartpole

```eval_rst
.. figure:: ./images/tf.png
        :scale: 10%
.. literalinclude:: ../../examples/tf/erwr_cartpole.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

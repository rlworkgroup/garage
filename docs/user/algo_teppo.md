# Proximal Policy Optimization with Task Embedding (TEPPO)


```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Learning Skill Embeddings for Transferable Robot Skills :cite:`hausman2018learning`
   * - **Framework(s)**
     - .. figure:: ./images/tf.png
        :scale: 20%
        :class: no-scaled-link

        Tensorflow
   * - **API Reference**
     - `garage.tf.algos.TEPPO <../_autoapi/garage/torch/algos/index.html#garage.tf.algos.TEPPO>`_
   * - **Code**
     - `garage/tf/algos/te_ppo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/te_ppo.py>`_
   * - **Examples**
     - :ref:`te_ppo_metaworld_mt1_push`, :ref:`te_ppo_metaworld_mt10`, :ref:`te_ppo_metaworld_mt50`, :ref:`te_ppo_point`
```

Proximal Policy Optimization Algorithms (PPO) is a family of policy gradient methods which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. TEPPO parameterizes the PPO policy via a shared skill embedding space.

## Default Parameters

```py
discount=0.99,
gae_lambda=0.98,
lr_clip_range=0.01,
max_kl_step=0.01,
policy_ent_coeff=1e-3,
encoder_ent_coeff=1e-3,
inference_ce_coeff=1e-3
```

## Examples

### te_ppo_metaworld_mt1_push

```eval_rst
.. literalinclude:: ../../examples/tf/te_ppo_metaworld_mt1_push.py
```

### te_ppo_metaworld_mt10

```eval_rst
.. literalinclude:: ../../examples/tf/te_ppo_metaworld_mt10.py
```

### te_ppo_metaworld_mt50

```eval_rst
.. literalinclude:: ../../examples/tf/te_ppo_metaworld_mt50.py
```

### te_ppo_point

```eval_rst
.. literalinclude:: ../../examples/tf/te_ppo_point.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

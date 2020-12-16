<!-- markdownlint-disable no-inline-html -->
# RL<sup>2</sup>

```eval_rst
+-------------------+----------------------------------------------------------------------------------------------------------+
| **Paper**         | RL\ :sup:`2` : Fast Reinforcement Learning via Slow Reinforcement Learning :cite:`duan2016rl`            |
+-------------------+----------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/tf.png                                                                              |
|                   |    :scale: 20%                                                                                           |
|                   |    :class: no-scaled-link                                                                                |
|                   |                                                                                                          |
|                   |    TensorFlow                                                                                            |
+-------------------+----------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.tf.algos.RL2 <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.RL2>`_                      |
+-------------------+----------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/tf/algos/rl2.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2.py>`_ |
+-------------------+----------------------------------------------------------------------------------------------------------+
```

When sampling for RL<sup>2</sup>, there are more than one environments to be
sampled from. In the original implementation, within each trial, all episodes
sampled will be concatenated into one single episode, and fed to the
inner algorithm. Thus, returns and advantages are calculated across the
episode.

## RL<sup>2</sup>PPO

Proximal Policy Optimization specific for RL<sup>2</sup>. Below are some
examples of running RL<sup>2</sup> in different environments.

### rl2_ppo_halfcheetah

```eval_rst
.. literalinclude:: ../../examples/tf/rl2_ppo_halfcheetah.py
```

### rl2_ppo_metaworld_ml10

```eval_rst
.. literalinclude:: ../../examples/tf/rl2_ppo_metaworld_ml10.py
```

### rl2_ppo_halfcheetah_meta_test

```eval_rst
.. literalinclude:: ../../examples/tf/rl2_ppo_halfcheetah_meta_test.py
```

## RL<sup>2</sup>TRPO

Trust Region Policy Optimization specific for RL<sup>2</sup>.

## rl2_trpo_halfcheetah

```eval_rst
.. literalinclude:: ../../examples/tf/rl2_trpo_halfcheetah.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

<!-- markdownlint-disable no-inline-html -->
# RL<sup>2</sup>

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - RL\ :sup:`2` : Fast Reinforcement Learning via Slow Reinforcement
       Learning :cite:`duan2016rl`
   * - **Framework(s)**
     - .. figure:: ./images/tensorflow.png
        :scale: 20%

        TensorFlow
   * - **API Reference**
     - `garage.tf.algos.RL2 <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.RL2>`_
   * - **Code**
     - `garage/tf/algos/rl2.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2.py>`_
   * - **Examples**
     - `examples <algo_rl2.html#examples>`_
```

When sampling for RL<sup>2</sup>, there are more than one environments to be
sampled from. In the original implementation, within each trial, all episodes
sampled will be concatenated into one single episode, and fed to the
inner algorithm. Thus, returns and advantages are calculated across the
episode.

User should not instantiate RL2 directly. Currently garage supports [PPO](https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2ppo.py)
and [TRPO](https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/rl2trpo.py)
as the inner algorithm.

## Examples

```eval_rst
.. literalinclude:: ../../examples/tf/rl2_ppo_halfcheetah.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

# Soft Actor-Critic

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Soft Actor-Critic Algorithms and Applications :cite:`haarnoja2018soft`
   * - **Framework(s)**
     - .. figure:: ./images/pytorch.png
        :scale: 10%

        PyTorch
   * - **API Reference**
     - `garage.torch.algos.BC <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.SAC>`_
   * - **Code**
     - `garage/torch/algos/bc.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/sac.py>`_
   * - **Examples**
     - `examples <algo_sac.html#examples>`_
```

Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic policy in
an off-policy way, forming a bridge between stochastic policy optimization and
DDPG-style approaches. A central feature of SAC is entropy regularization. The
policy is trained to maximize a trade-off between expected return and entropy,
a measure of randomness in the policy. This has a close connection to the
exploration-exploitation trade-off: increasing entropy results in more
exploration, which can accelerate learning later on. It can also prevent the
policy from prematurely converging to a bad local optimum.

## Default Parameters

```python
gradient_steps_per_itr=1000
max_episode_length_eval=1000
min_buffer_size=1e4
target_update_tau=5e-3
discount=0.99
buffer_batch_size=256
reward_scale=1.
steps_per_epoch=1
```

## Examples

```eval_rst
.. literalinclude:: ../../examples/torch/sac_half_cheetah_batch.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

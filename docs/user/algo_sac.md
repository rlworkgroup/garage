# Soft Actor-Critic (SAC)

```eval_rst
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **Action Space**  | Continuous                                                                                                     |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **Paper**         | Soft Actor-Critic Algorithms and Applications :cite:`haarnoja2018soft`                                         |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                               |
|                   |    :scale: 10%                                                                                                 |
|                   |    :class: no-scaled-link                                                                                      |
|                   |                                                                                                                |
|                   |    PyTorch                                                                                                     |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.SAC <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.SAC>`_                   |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/sac.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/sac.py>`_ |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| **Examples**      | `examples <algo_sac.html#examples>`_                                                                           |
+-------------------+----------------------------------------------------------------------------------------------------------------+
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
initial_log_entropy=0.
discount=0.99
buffer_batch_size=64
min_buffer_size=int(1e4)
target_update_tau=5e-3
policy_lr=3e-4
qf_lr=3e-4
reward_scale=1.0
optimizer=torch.optim.Adam
steps_per_epoch=1
num_evaluation_episodes=10
```

## Examples

```eval_rst
.. literalinclude:: ../../examples/torch/sac_half_cheetah_batch.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*

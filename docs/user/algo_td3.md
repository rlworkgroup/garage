# Twin Delayed Deep Deterministic (TD3)

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Addressing Function Approximation Error in Actor-Critic Methods :cite:`Fujimoto2018AddressingFA`
   * - **Framework(s)**
     - .. figure:: ./images/pytorch.png
        :scale: 10%
        :class: no-scaled-link

        PyTorch
       .. figure:: ./images/tf.png
        :scale: 20%
        :class: no-scaled-link

        Tensorflow
   * - **API Reference**
     - `garage.tf.algos.TD3 <https://garage.readthedocs.io/en/latest/_autoapi/garage/tf/algos/index.html#garage.tf.algos.TD3>`_
   * - **Code**
     - `garage/tf/algos/td3.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/td3.py>`_
   * - **Examples**
     - :ref:`td3_pendulum_tf`
   * - **Benchmarks**
     - :ref:`td3_garage_tf`
```

Twin Delayed Deep Deterministic (TD3) is an alogrithm motivated by Double Q-learning and built by taking the minimum value between two critic networks to prevent the overestimation of the value function. Garage's implementation is based on the paper's approach, which includes clipped Double Q-learning, delayed update of target and policy networks as well as target policy smoothing.

## Default Parameters

```py
target_update_tau=0.01,
policy_lr=1e-4,
qf_lr=1e-3,
discount=0.99,
exploration_policy_sigma=0.2,
exploration_policy_clip=0.5,
actor_update_period=2,
```

## Examples

### td3_pendulum_tf

```eval_rst
.. literalinclude:: ../../examples/tf/td3_pendulum.py
```

## Benchmarks

### Benchmarks Results

![TD3 TF HalfCheetah-v2](images/td3_tf_HalfCheetah-v2.png) ![TD3 TF Hopper-v2](images/td3_tf_Hopper-v2.png)
![TD3 TF InvertedDoublePendulum-v2](images/td3_tf_InvertedDoublePendulum-v2.png) ![TD3 TF InvertedPendulum-v2](images/td3_tf_InvertedPendulum-v2.png)
![TD3 TF Swimmer-v2](images/td3_tf_Swimmer-v2.png)

### td3_garage_tf

```eval_rst
.. literalinclude:: ../../benchmarks/src/garage_benchmarks/experiments/algos/td3_garage_tf.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*

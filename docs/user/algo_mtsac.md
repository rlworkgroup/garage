# Multi-Task Soft Actor-Critic (MT-SAC)

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Action Space**
     - Continuous
   * - **Framework(s)**
     - .. figure:: ./images/pytorch.png
        :scale: 10%

        PyTorch
   * - **API Reference**
     - `garage.torch.algos.MTSAC <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.MTSAC>`_
   * - **Code**
     - `garage/torch/algos/mtsac.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/mtsac.py>`_
   * - **Examples**
     - :ref:`mtsac_metaworld_ml1_pick_place`, :ref:`mtsac_metaworld_mt10`, :ref:`mtsac_metaworld_mt50`
```

The Multi-Task Soft Actor-Critic (MT-SAC) algorithm is the same as the [Soft Actor Critic (SAC)](algo_sac) algorithm, except for a small change called "disentangled alphas". Alpha is the entropy coefficient that is used to control exploration of the agent/policy. Disentangling alphas refers to having a separate alpha coefficients for every task learned by the policy. The alphas are accessed by using a one-hot encoding of an id that is assigned to each task.


## Default Parameters

```python
initial_log_entropy=0.,
discount=0.99,
buffer_batch_size=64,
min_buffer_size=int(1e4),
target_update_tau=5e-3,
policy_lr=3e-4,
qf_lr=3e-4,
reward_scale=1.0,
optimizer=torch.optim.Adam,
steps_per_epoch=1,
num_evaluation_episodes=5,
use_deterministic_evaluation=True,
```

## Examples

### mtsac_metaworld_ml1_pick_place
```eval_rst
.. literalinclude:: ../../examples/torch/mtsac_metaworld_ml1_pick_place.py
```

### mtsac_metaworld_mt10
```eval_rst
.. literalinclude:: ../../examples/torch/mtsac_metaworld_mt10.py
```

### mtsac_metaworld_mt50
```eval_rst
.. literalinclude:: ../../examples/torch/mtsac_metaworld_mt10.py
```

----

*This page was authored by Nicole Shin Ying Ng ([@nicolengsy](https://github.com/nicolengsy)).*

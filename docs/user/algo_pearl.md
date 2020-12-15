# Probablistic Embeddings for Actor-Critic Reinforcement Learning (PEARL)

```eval_rst
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
| **Paper**         | Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables :cite:`rakelly2019efficient`        |
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
| **Framework(s)**  | .. figure:: ./images/pytorch.png                                                                                         |
|                   |    :scale: 10%                                                                                                           |
|                   |    :class: no-scaled-link                                                                                                |
|                   |                                                                                                                          |
|                   |    PyTorch                                                                                                               |
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
| **API Reference** | `garage.torch.algos.PEARL <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PEARL>`_                         |
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
| **Code**          | `garage/torch/algos/pearl.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/pearl.py>`_       |
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
| **Examples**      | :ref:`pearl_half_cheetah_vel`, :ref:`pearl_metaworld_ml1_push`, :ref:`pearl_metaworld_ml10`, :ref:`pearl_metaworld_ml45` |
+-------------------+--------------------------------------------------------------------------------------------------------------------------+
```

PEARL, which stands for Probablistic Embeddings for Actor-Critic Reinforcement Learning, is an off-policy meta-RL algorithm. It is built on top of SAC using two Q-functions and a value function with an addition of an inference network that estimates the posterior `𝑞(𝑧‖𝑐)`. The policy is conditioned on the latent variable `Z` in order to adpat its behavior to specific tasks.

## Default Parameters

```python
batch_size=256,
embedding_batch_size=100,
embedding_mini_batch_size=100,
encoder_hidden_size=200,
latent_size=5,
max_episode_length=200,
meta_batch_size=16,
net_size=300,
num_epochs=500,
num_train_tasks=100,
num_test_tasks=30,
num_steps_per_epoch=2000,
num_initial_steps=2000,
num_tasks_sample=5,
num_steps_prior=400,
num_extra_rl_steps_posterior=600,
reward_scale=5.
```

## Examples

### pearl_half_cheetah_vel

```eval_rst
.. literalinclude:: ../../examples/torch/pearl_half_cheetah_vel.py
```

### pearl_metaworld_ml1_push

```eval_rst
.. literalinclude:: ../../examples/torch/pearl_metaworld_ml1_push.py
```

### pearl_metaworld_ml10

```eval_rst
.. literalinclude:: ../../examples/torch/pearl_metaworld_ml10.py
```

### pearl_metaworld_ml45

```eval_rst
.. literalinclude:: ../../examples/torch/pearl_metaworld_ml45.py
```

## References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*

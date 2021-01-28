# garage

garage is a toolkit for developing and evaluating reinforcement learning
algorithms, and an accompanying library of state-of-the-art implementations
built using that toolkit.

garage is a work in progress, input is welcome. The available documentation is
limited, but rapidly growing.

## User Guide

The garage user guide explains how to install garage, how to run experiments,
and how to implement new MDPs and new algorithms.

```eval_rst
.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   user/installation
   user/get_started

.. toctree::
   :maxdepth: 2
   :caption: Usage Guide (How-To)

   user/experiments
   user/pixel_observations
   user/monitor_experiments_with_tensorboard
   user/training_a_policy
   user/save_load_resume_exp
   user/reuse_garage_policy
   user/use_pretrained_network_to_start_new_experiment
   user/docker
   user/ensure_your_experiments_are_reproducible
   user/meta_multi_task_rl_exp.md
   user/max_resource_usage
   user/cluster_setup

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   user/implement_env
   user/implement_algo
   user/custom_worker

.. toctree::
   :maxdepth: 2
   :caption: Algorithms and Methods

   BC <user/algo_bc>
   CEM <user/algo_cem>
   DDPG <user/algo_ddpg>
   ERWR <user/algo_erwr>
   MAML <user/algo_maml>
   MT-PPO <user/algo_mtppo>
   MT-SAC <user/algo_mtsac>
   MT-TRPO <user/algo_mttrpo>
   PEARL <user/algo_pearl>
   PPO <user/algo_ppo>
   RL2 <user/algo_rl2>
   SAC <user/algo_sac>
   TD3 <user/algo_td3>
   TRPO <user/algo_trpo>
   REINFORCE <user/algo_vpg>
   DQN <user/algo_dqn>

.. toctree::
   :maxdepth: 2
   :caption: Reference Guide

   user/environment
   user/environment_libraries
   user/concept_experiment
   user/sampling
   user/logging_plotting

.. toctree::
   :maxdepth: 2
   :caption: Development Guide

   user/setting_up_your_development_environment
   user/testing
   user/benchmarking
   user/writing_documentation
   user/git_workflow
   user/preparing_a_pr
   user/docker_dev

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   _autoapi/garage/index
   _autoapi/garage/envs/index
   _autoapi/garage/experiment/index
   _autoapi/garage/np/index
   _autoapi/garage/plotter/index
   _autoapi/garage/replay_buffer/index
   _autoapi/garage/sampler/index
   _autoapi/garage/tf/index
   _autoapi/garage/torch/index
```

## Citing garage

If you use garage for academic research, please cite the repository using the
following BibTeX entry. You should update the `commit` field with the commit or
release tag your publication uses.

```
@misc{garage,
  author = {The garage contributors},
  title = {Garage: A toolkit for reproducible reinforcement learning research},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rlworkgroup/garage}},
  commit = {ebd7800430b0212c3ffcf78fd3ec26b22097c371}
```

## Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

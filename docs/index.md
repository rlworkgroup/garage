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

.. toctree::
   :maxdepth: 2
   :caption: Usage Guide (How-To)

   user/experiments
   user/pixel_observations
   user/monitor_experiments_with_tensorboard
   user/training_a_policy
   user/save_load_resume_exp
   user/reuse_garage_policy
   user/docker
   user/ensure_your_experiments_are_reproducible

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   user/implement_env
   user/implement_algo

.. toctree::
   :maxdepth: 2
   :caption: Algorithms and Methods

   user/algo_bc

.. toctree::
   :maxdepth: 2
   :caption: Development Guide

   user/setting_up_your_development_environment
   user/testing
   user/benchmarking
   user/writing_documentation

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   _autoapi/garage/index
   _autoapi/garage/envs/index
   _autoapi/garage/experiment/index
   _autoapi/garage/misc/index
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

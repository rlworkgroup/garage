# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2019.02.2](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.2)

### Fixed
- Bug in entropy regularization in TensorFlow PPO/TRPO ([#579](https://github.com/rlworkgroup/garage/pull/579))
- Bug in which advantage normalization was broken for recurrent policies ([#626](https://github.com/rlworkgroup/garage/pull/626))
- Bug in `examples/sim_policy.py` ([#691](https://github.com/rlworkgroup/garage/pull/691))
- Bug in `FiniteDifferenceHvp` ([#745](https://github.com/rlworkgroup/garage/pull/745))

## [2019.02.1](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.1)
### Fixed
- Fix overhead in GaussianMLPRegressor by optionally creating assign operations ([#622](https://github.com/rlworkgroup/garage/pull/622))


## [2019.02.0](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.0)

### Added
- Epsilon-greedy exploration strategy, DiscreteMLPModel, and
  QFunctionDerivedPolicy (all needed by DQN)
- Base Model class for TensorFlow-based primitives
- Dump plots generated with matplotlib to TensorBoard
- Relative Entropy Policy Search (REPS) algorithm
- GaussianConvBaseline and GaussianConvRegressor primitives
- New Dockerfiles, docker-compose files, and Makefiles for running garage using
  Docker
- Vanilla policy gradient loss to NPO
- Truncated Natural Policy Gradient (TNPG) algorithm for TensorFlow
- Episodic Reward Weighted Regression (ERWR) algorithm for TensorFlow
- gym.Env wrappers used for pixel environments
- Convolutional Neural Network primitive

### Changed
- Move dependencies from environment.yml to setup.py
- Update dependencies:
  - tensorflow-probability to 0.5.x
  - dm_control to commit 92f9913
  - TensorFlow to 1.12
  - MuJoCo to 2.0
  - gym to 0.10.11
- Move dm_control tests into the unit test tree
- Use GitHub standard .gitignore
- Improve the implementation of RandomizedEnv (Dynamics Randomization)
- Decouple TensorBoard from the logger
- Move files from garage/misc/instrument to garage/experiment
- setup.py to be canonical in format and use automatic versioning

### Removed
- Move some garage subpackages into their own repositories:
  - garage.viskit to [rlworkgroup/viskit](https://github.com/rlworkgroup/viskit)
  - garage.spaces to [rlworkgroup/akro](https://github.com/rlworkgroup/akro)
- Remove Theano backend, algorithms, and dependencies
- Custom environments which duplicated [openai/gym](https://github.com/openai/gym)
- Some dead files from garage/misc (meta.py and viewer2d.py)
- Remove all code coverage tracking providers except CodeCov

### Fixed
- Clean up warnings in the test suite
- Pickling bug in GaussianMLPolicyWithModel
- Namescope in LbfgsOptimizer
- Correctly sample paths in OffPolicyVectorizedSampler
- Implementation bugs in tf/VPG
- Bug when importing Box
- Bug in test_benchmark_her

## [2018.10.1](https://github.com/rlworkgroup/garage/releases/tag/v2018.10.1)

### Fixed
- Avoid importing Theano when using the TensorFlow branch
- Avoid importing MuJoCo when not required
- Implementation bugs in tf/VPG
- Bug when importing Box
- Bug in test_benchmark_her
- Bug in the CI scripts which produced false positives

## [2018.10.0](https://github.com/rlworkgroup/garage/releases/tag/v2018.10.1)

### Added
- PPO and DDPG for the TensorFlow branch
- HER for DDPG
- Recurrent Neural Network policy support for NPO, PPO and TRPO
- Base class for ReplayBuffer, and two implementations: SimpleReplayBuffer
  and HerReplayBuffer
- Sampler classes OffPolicyVectorizedSampler and OnPolicyVectorizedSampler
- Base class for offline policies OffPolicyRLAlgorithm
- Benchmark tests for TRPO, PPO and DDPG to compare their performance with
  those produced by OpenAI Baselines
- Dynamics randomization for MuJoCo environments
- Support for dm_control environments
- DictSpace support for garage environments
- PEP8 checks enforced in the codebase
- Support for Python imports: maintain correct ordering and remove unused
  imports or import errors
- Test on TravisCI using Docker images for managing dependencies
- Testing code reorganized
- Code Coverage measurement with codecov
- Pre-commit hooks to enforce PEP8 and to verify imports and commit messages,
  which are also applied in the Travis CI verification
- Docstring verification for added files that are not in the test branch or
  moved files
- TensorBoard support for all key-value/log_tabular calls, plus support for
  logging distributions
- Variable and name scope for symbolic operations in TensorFlow
- Top-level base Space class for garage
- Asynchronous plotting for Theano and Tensorflow
- GPU support for Theano

### Changed
- Rename rllab to garage, including all the rllab references in the packages
  and modules inside the project
- Rename run_experiment_lite to run_experiment
- The file cma_es_lib.py was replaced by the pycma library available on PyPI
- Move the contrib package to garage.contrib
- Move Theano-dependent code to garage.theano
- Move all code from sandbox.rocky.tf to garage.tf
- Update several dependencies, mainly:
  - Python to 3.6.6
  - TensorFlow to 1.9
  - Theano to 1.0.2
  - mujoco-py to 1.50.1
  - gym to 0.10.8
- Transfer various dependencies from conda to pip
- Separate example script files in the Theano and TensorFlow branch
- Update LICENSE, CONTRIBUTING.md and .gitignore
- Use convenience imports, that is, import classes and functions that share the
  same or similar name to its module in the corresponding `__init__.py` file of
  their package
- Replace ProxyEnv with gym.Wrapper
- Update installation scripts for Linux and macOS

### Removed
- All unused imports in the Python files
- Unused packages from environment.yml
- The files under rllab.mujoco_py were removed to use the pip release instead
- Empty `__init__.py` files
- The environment class defined by rllab.envs.Env was not imported to garage
  and the environment defined by gym.Env is used now

### Fixed
- Sleeping processes produced by the parallel sampler. NOTE: although the
  frequency of this issue has been reduced, our tests in TravisCI occasionally
  detect the issue and currently it seems to be an issue with re-entrant locks
  and multiprocessing in Python.

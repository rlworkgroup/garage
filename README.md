[![Docs](https://readthedocs.org/projects/garage/badge)](http://garage.readthedocs.org/en/latest/)
[![Build Status](https://travis-ci.com/rlworkgroup/garage.svg?branch=master)](https://travis-ci.com/rlworkgroup/garage)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/garage/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/rlworkgroup/garage/branch/master/graph/badge.svg)](https://codecov.io/gh/rlworkgroup/garage)
[![PyPI version](https://badge.fury.io/py/garage.svg)](https://badge.fury.io/py/garage)

# garage

garage is a toolkit for developing and evaluating reinforcement learning algorithms, and an accompanying library of state-of-the-art implementations built using that toolkit.

The toolkit provides wide range of modular tools for implementing RL algorithms, including:
* Composable neural network models
* Replay buffers
* High-performance samplers
* An expressive experiment definition interface
* Tools for reproducibility (e.g. set a global random seed which all components respect)
* Logging to many outputs, including TensorBoard
* Reliable experiment checkpointing and resuming
* Environment interfaces for many popular benchmark suites
* Supporting for running garage in diverse environments, including always up-to-date Docker containers

See the [latest documentation](https://garage.readthedocs.org/en/latest/) for getting started instructions and detailed APIs.

## Installation
```
pip install garage
```

## Algorithms
The table below summarizes the algorithms available in garage.

| Algorithm              | Framework(s)        |
| ---------------------- | ------------------- |
| CEM                    | numpy               |
| CMA-ES                 | numpy               |
| REINFORCE (a.k.a. VPG) | PyTorch, TensorFlow |
| DDPG                   | PyTorch, TensorFlow |
| DQN                    | TensorFlow          |
| DDQN                   | TensorFlow          |
| ERWR                   | TensorFlow          |
| NPO                    | TensorFlow          |
| PPO                    | TensorFlow          |
| REPS                   | TensorFlow          |
| TD3                    | TensorFlow          |
| TNPG                   | TensorFlow          |
| TRPO                   | TensorFlow          |

## Supported Tools and Frameworks
garage supports Python 3.5+.

We currently support [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) for implementing the neural network portions of RL algorithms, and additions of new framework support are always welcome. PyTorch modules can be found in the package [`garage.torch`](https://github.com/rlworkgroup/garage/tree/master/src/garage/torch) and TensorFlow modules can be found in the package [`garage.tf`](https://github.com/rlworkgroup/garage/tree/master/src/garage/tf). Algorithms which do not require neural networks are found in the package [`garage.np`](https://github.com/rlworkgroup/garage/tree/master/src/garage/np).

The package is available for download on PyPI, and we ensure that it installs successfully into environments defined using [conda](https://docs.conda.io/en/latest/), [Pipenv](https://pipenv.readthedocs.io/en/latest/), and [virtualenv](https://virtualenv.pypa.io/en/latest/).

All components use the popular [`gym.Env`](https://github.com/openai/gym) interface for RL environments.

## Testing
The most important feature of garage is its comprehensive automated unit test and benchmarking suite, which helps ensure that the algorithms and modules in garage maintain state-of-the-art performance as the software changes.

Our testing strategy has three pillars:

* **Automation:**
  We use continuous integration to test all modules and algorithms in garage before adding any change. The full installation and test suite is also run nightly, to detect regressions.
* **Acceptance Testing:**
  Any commit which might change the performance of an algorithm is subjected to comprehensive benchmarks on the relevant algorithms before it is merged
* **Benchmarks and Monitoring:**
  We benchmark the full suite of algorithms against their relevant benchmarks and widely-used implementations regularly, to detect regressions and improvements we may have missed.

## Supported Releases
| Release | Build Status | Last date of support |
| ------- | ------------ | -------------------- |
| [v2019.10](https://github.com/rlworkgroup/garage/releases/tag/v2019.10.0) | [![Build Status](https://travis-ci.com/rlworkgroup/garage.svg?branch=release-2019.10)](https://travis-ci.com/rlworkgroup/garage) | June 30th, 2020 |

Garage releases a new stable version approximately every 4 months, in February, June, and October. Maintenance releases have a stable API and dependency tree, and receive bug fixes and critical improvements but not new features. We currently support each release for a window of 8 months.

## Citing garage

If you use garage for academic research, please cite the repository using the following BibTeX entry. You should update the `commit` field with the commit or release tag your publication uses.

```latex
@misc{garage,
 author = {The garage contributors},
 title = {Garage: A toolkit for reproducible reinforcement learning research},
 year = {2019},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/rlworkgroup/garage}},
 commit = {be070842071f736eb24f28e4b902a9f144f5c97b}
}
```

## Credits
The original code for garage was adopted from predecessor project called [rllab](https://github.com/rll/rllab). The garage project is grateful for the contributions of the original rllab authors, and hopes to continue advancing the state of reproducibility in RL research in the same spirit.

rllab was developed by Rocky Duan (UC Berkeley/OpenAI), Peter Chen (UC Berkeley), Rein Houthooft (UC Berkeley/OpenAI), John Schulman (UC Berkeley/OpenAI), and Pieter Abbeel (UC Berkeley/OpenAI).

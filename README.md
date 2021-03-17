[![Docs](https://readthedocs.org/projects/garage/badge)](http://garage.readthedocs.org/en/latest/)
[![Garage CI](https://github.com/rlworkgroup/garage/workflows/Garage%20CI/badge.svg?event=schedule)](https://github.com/rlworkgroup/garage/actions?query=workflow%3A%22Garage+CI%22)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/garage/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/rlworkgroup/garage/branch/master/graph/badge.svg)](https://codecov.io/gh/rlworkgroup/garage)
[![PyPI version](https://badge.fury.io/py/garage.svg)](https://badge.fury.io/py/garage)

# garage

garage is a toolkit for developing and evaluating reinforcement learning
algorithms, and an accompanying library of state-of-the-art implementations
built using that toolkit.

The toolkit provides wide range of modular tools for implementing RL algorithms,
including:

* Composable neural network models
* Replay buffers
* High-performance samplers
* An expressive experiment definition interface
* Tools for reproducibility (e.g. set a global random seed which all components
  respect)
* Logging to many outputs, including TensorBoard
* Reliable experiment checkpointing and resuming
* Environment interfaces for many popular benchmark suites
* Supporting for running garage in diverse environments, including always
  up-to-date Docker containers

See the [latest documentation](https://garage.readthedocs.org/en/latest/) for
getting started instructions and detailed APIs.

## Installation

```
pip install --user garage
```

## Examples

Starting from version v2020.10.0, garage comes packaged with examples. To get a
list of examples, run:

```
garage examples
```

You can also run `garage examples --help`, or visit
[the documentation](https://garage.readthedocs.io/en/latest/user/get_started.html#running-examples)
for even more details.

## Join the Community

**Join the [garage-announce mailing list](https://groups.google.com/forum/#!forum/garage-announce/join)**
for infrequent updates (<1/mo.) on the status of the project and new releases.

Need some help? Want to ask garage is right for your project? Have a question
which is not quite a bug and not quite a feature request?

**Join the community Slack** by filling out
[this Google Form](https://docs.google.com/forms/d/e/1FAIpQLSf4AXRIbA1cLGjku4lIRQ6btStWPeIMeG3J17i4_FhFQU8X0g/viewform).

## Algorithms

The table below summarizes the algorithms available in garage.

| Algorithm              | Framework(s)        |
| ---------------------- | ------------------- |
| CEM                    | numpy               |
| CMA-ES                 | numpy               |
| REINFORCE (a.k.a. VPG) | PyTorch, TensorFlow |
| DDPG                   | PyTorch, TensorFlow |
| DQN                    | PyTorch, TensorFlow |
| DDQN                   | PyTorch, TensorFlow |
| ERWR                   | TensorFlow          |
| NPO                    | TensorFlow          |
| PPO                    | PyTorch, TensorFlow |
| REPS                   | TensorFlow          |
| TD3                    | PyTorch, TensorFlow |
| TNPG                   | TensorFlow          |
| TRPO                   | PyTorch, TensorFlow |
| MAML                   | PyTorch             |
| RL2                    | TensorFlow          |
| PEARL                  | PyTorch             |
| SAC                    | PyTorch             |
| MTSAC                  | PyTorch             |
| MTPPO                  | PyTorch, TensorFlow |
| MTTRPO                 | PyTorch, TensorFlow |
| Task Embedding         | TensorFlow          |
| Behavioral Cloning     | PyTorch             |

## Supported Tools and Frameworks

garage requires Python 3.6+. If you need Python 3.5 support, the last garage
release to support Python 3.5 was
[v2020.06](https://github.com/rlworkgroup/garage/releases/tag/v2020.06.0).

The package is tested on Ubuntu 18.04. It is also known to run on Ubuntu 16.04,
18.04, and 20.04, and recent versions of macOS using Homebrew. Windows users can
install garage via WSL, or by making use of the Docker containers.

We currently support [PyTorch](https://pytorch.org/) and
[TensorFlow](https://www.tensorflow.org/) for implementing the neural network
portions of RL algorithms, and additions of new framework support are always
welcome. PyTorch modules can be found in the package
[`garage.torch`](https://github.com/rlworkgroup/garage/tree/master/src/garage/torch)
and TensorFlow modules can be found in the package
[`garage.tf`](https://github.com/rlworkgroup/garage/tree/master/src/garage/tf).
Algorithms which do not require neural networks are found in the package
[`garage.np`](https://github.com/rlworkgroup/garage/tree/master/src/garage/np).

The package is available for download on PyPI, and we ensure that it installs
successfully into environments defined using
[conda](https://docs.conda.io/en/latest/),
[Pipenv](https://pipenv.readthedocs.io/en/latest/), and
[virtualenv](https://virtualenv.pypa.io/en/latest/).

## Testing

The most important feature of garage is its comprehensive automated unit test
and benchmarking suite, which helps ensure that the algorithms and modules in
garage maintain state-of-the-art performance as the software changes.

Our testing strategy has three pillars:

* **Automation:**
  We use continuous integration to test all modules and algorithms in garage
  before adding any change. The full installation and test suite is also run
  nightly, to detect regressions.
* **Acceptance Testing:**
  Any commit which might change the performance of an algorithm is subjected to
  comprehensive benchmarks on the relevant algorithms before it is merged
* **Benchmarks and Monitoring:**
  We benchmark the full suite of algorithms against their relevant benchmarks
  and widely-used implementations regularly, to detect regressions and
  improvements we may have missed.

## Supported Releases

| Release | Build Status | Last date of support |
| ------- | ------------ | -------------------- |
| [v2021.03](https://github.com/rlworkgroup/garage/releases/tag/v2021.03.0) | [![Garage CI Release-2021.03](https://github.com/rlworkgroup/garage/workflows/Garage%20CI%20Release-2021.03/badge.svg)](https://github.com/rlworkgroup/garage/actions?query=workflow%3A%22Garage+CI+Release-2021.03%22) | May 31st, 2021 |

Maintenance releases have a stable API and dependency tree,
and receive bug fixes and critical improvements but not new features. We
currently support each release for a window of 2 months.

## Citing garage

If you use garage for academic research, please cite the repository using the
following BibTeX entry. You should update the `commit` field with the commit or
release tag your publication uses.

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

The earliest code for garage was adopted from predecessor project called
[rllab](https://github.com/rll/rllab). The garage project is grateful for the
contributions of the original rllab authors, and hopes to continue advancing the
state of reproducibility in RL research in the same spirit. garage has
previously been supported by the Amazon Research Award "Watch, Practice, Learn,
Do: Unsupervised Learning of Robust and Composable Robot Motion Skills by Fusing
Expert Demonstrations with Robot Experience."

---
<p align="center" style="align-items:center; display:inline-block">Made with &#10084; &nbsp;at <a href="https://robotics.usc.edu/resl/" target="_blank"><img align="absmiddle" src="https://github.com/rlworkgroup/garage/blob/master/docs/_static/resl_logo.png?raw=true" height="60px"></a> and &nbsp;<a href="https://viterbischool.usc.edu/" target="_blank"><img align="absmiddle" src="https://github.com/rlworkgroup/garage/blob/master/docs/_static/viterbi_logo.png?raw=true" height="30px"></a></p>

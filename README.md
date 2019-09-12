[![Docs](https://readthedocs.org/projects/garage/badge)](http://garage.readthedocs.org/en/latest/)
[![Build Status](https://travis-ci.com/rlworkgroup/garage.svg?branch=master)](https://travis-ci.com/rlworkgroup/garage)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/garage/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/rlworkgroup/garage/branch/master/graph/badge.svg)](https://codecov.io/gh/rlworkgroup/garage)
[![PyPI version](https://badge.fury.io/py/garage.svg)](https://badge.fury.io/py/garage)

# garage

garage is a framework for developing and evaluating reinforcement learning algorithms. It includes a wide range of continuous control tasks plus implementations of algorithms.

garage is fully compatible with [OpenAI Gym](https://gym.openai.com/). All garage environments implement gym.Env, so all garage components can also be used with any environment implementing gym.Env.

garage only officially supports Python 3.5+.

garage supports [TensorFlow](https://www.tensorflow.org/) for neural network frameworks. TensorFlow modules can be found under [garage/tf](https://github.com/rlworkgroup/garage/tree/master/garage/tf).

# Documentation

Documentation is available online at [https://garage.readthedocs.org/en/latest/](https://garage.readthedocs.org/en/latest/).

# Citing garage

If you use garage for academic research, you are highly encouraged to cite the following paper on the original rllab implementation:

- Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

# Credits
garage is based on a predecessor project called [rllab](https://github.com/rll/rllab). The garage project is grateful for the contributions of the original rllab authors, and hopes to continue advancing the state of reproducibility in RL research in the same spirit.

rllab was originally developed by Rocky Duan (UC Berkeley/OpenAI), Peter Chen (UC Berkeley), Rein Houthooft (UC Berkeley/OpenAI), John Schulman (UC Berkeley/OpenAI), and Pieter Abbeel (UC Berkeley/OpenAI).

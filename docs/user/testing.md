# Testing

In this section, we will talk about how to test different modules and algorithms in garage.

In garage, we use [pytest](https://docs.pytest.org/en/stable/getting-started.html) to carry out tests.

## Test modules

All test files are stored under `garage/tests` in the garage respository.

Test modules are organized in the same way as the garage main repository. Ideally, all main modules and files under `garage/src/garage` should be covered with test cases.

## Prerequisites

To begin testing, we suggest using the same Python manager environment i.e. `virtualenv`, `conda` as we develop on garage repository so that all packages and depencies required for testing are installed in the virutal environment.

Let's begin by activating the virtual environment:

```bash
# Conda
$ conda activate myenv

# Virtualenv
$ source myenv/bin/activate
```

Next, we need to install pytest. Generally, pytest should have already been installed upon garage installation.

To get started, run pytest as follows:

```bash
#!/bin/bash
$ pytest
```

If you haven't done so, please go ahead and check out our installation guide first. It is also recommened to check out official [pytest](https://docs.pytest.org/en/stable/getting-started.html) documentation.

## Basic Pytest Usage

We will start with a simple test. To be consistent with pytest requirement and garage modules, we name our test `.py` file with pre-fix `test_`.

Let's write a simple test case for the vpg algorithm `garage/src/garage/tf/algos/vpg.py`.

We begin by creating a file called `test_vpg.py` and put it under the tree `garage/tests/garage/tf/algos/`.

We want to test VPG in the cart pole environment. Hence, we create a unitest named `test_vpg_cartpole`. Inside the function, we define the environemnt, the policy and the baselien and feed them to the VPG algorithm and run the experiment. Finally, we make sure the return value is identical to our expectation by using `assert`.

Sometimes, we modele multiple tests into a class for modular and scalable structure. There is no need to subclass or anything, but make sure the prefix of the class starts with `Test`, otherwise the class will be skipped.

```python
# test_vpg.py

import pytest
... # import other dependencies

class TestVPG(...):
    def test_vpg_cartpole(self):
        ...
        env = GarageEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          ...)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=100,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 90

            env.close()
```

## Run tests

In general, we can start running tests simply by:

```bash
#!/bin/bash
$ pytest
```

However, in most of the cases, we simply don't have the time to test everything, plus Travis CI will take care of the  majority of tests upon the deployment workflow. We can use the following ways to carry out specific tests to make life easier.

### Specifying tests / selecting tests

Pytest supports several eays to run and select tests.

#### Run tests in a directory

Run a test on particular module(s) by specifying a directory path.

```bash
#!/bin/bash
$ pytest tests/garage/tf/algo/
```

#### Run tests in a module

Run a test on a particular module by specifying a file path.

```bash
#!/bin/bash
$ pytest tests/garage/tf/algo/test_ppo.py
```

#### Run tests by keyword expressions

Run tests by keyword expressions. This is useful for running particular test function(s).

```bash
#!/bin/bash
$ pytest -k test_ppo_pendulum_continuous_baseline

```

## Useful pytest commands

### Parametrized Test Functions

Parametrized test function is a  delightful solution to save us from tedious testing in same scenarios with different parameters. We can simply specify the name of the arguments that will be pass to the test function and a list of arguments corresponding to the names.

```python
import pytest

@pytest.mark.parametrize('filters, in_channels, strides', [
        (((32, (1, 1)),), (3, ), (1, )),
        (((32, (3, 3)),), (3, ), (1, )),
        (((32, (3, 3)),), (3, ), (2, )),
        (((32, (1, 1)), (64, (1, 1))), (3, 32), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (3, 32), (1, 1)),
        (((32, (3, 3)), (64, (3, 3))), (3, 32), (2, 2)),
    ])
def test_output_value(self, filters, in_channels, strides):
    model = CNNModel(filters=filters,
                     strides=strides,
                     name='cnn_model',
                     padding='VALID',
                     hidden_w_init=tf. constant_initialize(1,
                     hidden_nonlinearity=None)
    ...

```

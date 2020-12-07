# Testing

In this section, we will talk about how to test
different modules and algorithms in garage.

In garage, we use [pytest](https://docs.pytest.org/en/stable/getting-started.html)
to carry out tests.

## Test Modules

All test files are stored under `tests/` in the
garage respository.

Test modules are organized in the same way as the
garage main repository. Ideally, all main modules
and files under `src/garage/` should be covered
with test cases.

Test modules of garage are structured in the following ways:

- `tests/` contains all tests and supporting code
- `tests/garage` contains unit tests
- `tests/integration_tests` contains integration tests
- modules in `tests/` and `tests/fixtures` contain helper
  codes and fixtures which make writing good tests easier.

```bash
tests
├── fixtures
├── garage
├── integration_tests
├── helpers.py
└── (helper_files.py)
```

## Prerequisites

To begin testing, we suggest using the same
Python manager environment i.e. `virtualenv`, `conda`
as we develop on garage repository so that all
packages and dependencies required for testing
are installed in the virutal environment.

Let's begin by activating the virtual environment:

```bash
# Conda
conda activate myenv

# Virtualenv
source myenv/bin/activate
```

Next, we need to install pytest. Generally, pytest
should have already been installed upon garage's
`dev` installation. To install the garage environment
 for testing, you will need to install the
`garage[all,dev]` for dependencies such as pytest,
mujoco and so on.

Also, you may want to check out our installation guide
first before diving into tests. It is also recommened
to check out official
[pytest](https://docs.pytest.org/en/stable/getting-started.html) documentation.

```bash
cd path/to/garage/
pip install -e '.[all,dev]'
```

To get started, run pytest as follows:

```bash
pytest
```

## Basic Pytest Usage

Congrats! Now you're ready for testing garage!

We will start with a simple test. To be consistent
with pytest requirement and garage modules,
we name our test `.py` file with pre-fix `test_`.

Let's write a simple test case for the vpg
algorithm `src/garage/tf/algos/vpg.py`.

We begin by creating a file called `test_vpg.py` and put it under the tree `tests/garage/tf/algos/`.

We want to test VPG in the cart pole environment.
Hence, we create a unitest named `test_vpg_cartpole`.
Inside the function, we define the environemnt,
the policy and the baselien and feed them to the
VPG algorithm and run the experiment. Finally,
we make sure the return value is identical to our
expectation by using `assert`.

In new xunit-style tests, multiple tests are modeled
into a class for modular and scalable structure.
There is no need to subclass or anything,
but make sure the prefix of the class starts
with `Test`, otherwise the class will be skipped.
Note that it's not encouraged to use this style,
especially when a test doesn't require `setup_method` and `teardown_method`.

```python
# test_vpg.py

class TestVPG(...):
    def test_vpg_cartpole(self):
        ...
        env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          ...)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length.
                is_tf_worker=True)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       sampler=sampler,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env)

            last_avg_ret = trainer.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 90

            env.close()
```

## Run Tests

In general, we can start running tests simply by:

```bash
pytest
```

However, in most of the cases, we simply don't have
the time to test everything, plus Travis CI will
 take care of the  majority of tests upon the
  deployment workflow. We can use the following
   ways to carry out specific tests to make life easier.

### Specifying Tests / Selecting Tests

Pytest supports several ways to run and select tests.

#### Run tests in a directory

Run a test on particular module(s) by specifying
a directory path.

```bash
pytest tests/garage/tf/algo/
```

#### Run tests in a module

Run a test on a particular module by specifying a file path.

```bash
pytest tests/garage/tf/algo/test_ppo.py
```

#### Run tests by keyword expressions

Run tests by keyword expressions. This is useful for running particular test function(s).

```bash
pytest -k test_ppo_pendulum_continuous_baseline

```

## Useful Pytest Methods

Below are the pytest methods and functions that
we found helpful for testing garage.

### Setup and teardown methods

`setup_method` is called before every tests to
set up the test environment. It setups any state
tied to the execution of the given method in
a class.  `setup_method` is invoked for every
test method of a class.

`teardown_method` is called after every tests to
teardown any state that was previously setup
with a `setup_method`.

For details on `setup_method` and `teardown_method`, check [this](https://docs.pytest.org/en/2.8.7/xunit_setup.html).

```Python
class TestSampleClass:

    def setup_method(self):
        """Setup method which is called before every test."""
        ...

    def teardown_method(self):
        """Teardown method which is called after every test."""
        ...

```

### Parametrized test functions

Parametrized test function is a  delightful solution to
save us from tedious testing in same scenarios with
different parameters. We can simply specify the name
of the arguments that will be pass to the test function
and a list of arguments corresponding to the names.

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

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*

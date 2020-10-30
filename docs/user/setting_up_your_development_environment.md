# Setting Up Your Development Environment

In this section you will learn how to install garage and its dependencies in a
manner that allows for development. If you would like to contribute changes back
to garage, please also read `CONTRIBUTING.md`.

## Installing Dependencies

Generally speaking, system dependencies of garage are minimal, and likely already
installed. However, many of the environments garage is used with have additional
dependencies, and we provide "setup scripts" for installing those dependencies
and working around known problems on common platforms.

Garage's test suite requires *all dependencies* to be installed. Therefore, as a
developer, you'll need a MuJoCo license key. You can get one
[here](https://www.roboti.us/license.html).

For information on how to install the dependencies, check out our [Installation
Guide](installation.rst).

## Installing garage as an editable package

- pipenv:

```
cd path/to/garage/repo
pipenv --three
# --pre required because garage has some dependencies with verion numbers <1.0
pipenv install --pre -e '.[all,dev]'
```

- conda:

```
conda activate myenv
pip uninstall garage  # To ensure no existing install gets in the way
cd path/to/garage/repo
pip install -e '.[all,dev]'
```

- virtualenv:

```
source myenv/bin/activate
pip uninstall garage  # To ensure no existing install gets in the way
cd path/to/garage/repo
pip install -e '.[all,dev]'
```

## Verifying that your environment works

As a simple test of your installation, navigate to garage's root directory and
run `make docs`. This should build garage's documentation.

To ensure you have an editable version of garage, you could modify a file like
`tests/garage/np/algos/test_cem.py`, changing the assert to something false (in
this case, you could change the `>` to `<`). Then run `pytest tests/garage/np`
in Garage's root directory, and make sure that the test you modified fails.

----

_This page was authored by Hayden Shively
([@haydenshively](https://github.com/haydenshively))_

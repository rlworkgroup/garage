# Contributing to garage
We welcome all contributions to garage.

Use this guide to prepare your contribution.

## Pull requests
All contributions to the garage codebase are submitted via a GitHub pull request.

### Review process
To be submitted, a pull request must satisfy the following criteria:
1. Rebases cleanly on the `master` branch
1. Passes all continuous integration tests
1. Conforms to the git commit message [format](#commit-message-format)
1. Receives approval from another contributor
1. Receives approval from a maintainer (distinct from the contributor review)

These criteria may be satisfied in any order, but in practice your PR is unlikely to get attention from contributors until 1-3 are satisfied. Maintainer attention is a scarce resource, so generally maintainers wait for a review from a non-maintainer contributor before reviewing your PR.

## Preparing your repo to make contributions
After following the standard garage setup steps, make sure to run to install the pre-commit hooks into your repository. pre-commit helps streamline the pull request process by catching basic problems locally before they are checked by the CI.

To setup pre-commit in your repo:
```sh
# make sure your Python environment is activated, e.g.
# conda activate garage
# pipenv shell
# poetry shell
# source venv/bin/activate
pre-commit install -t pre-commit
pre-commit install -t pre-push
pre-commit install -t commit-msg
```

Once you've installed pre-commit, it will automatically run every time you type `git commit`.

## Code style
The Python code in garage conforms to the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard. Please read and understand it in detail.

### garage-specific Python style
These are garage-specific rules which are not part of the aforementioned style guides.
* Python package imports should be sorted alphabetically within their PEP8 groupings.

    The sorting is alphabetical from left to right, ignoring case and Python keywords (i.e. `import`, `from`, `as`). Notable exceptions apply in `__init__.py` files, where sometimes this rule will trigger a circular import.

* Prefer single-quoted strings (`'foo'`) over double-quoted strings (`"foo"`).

    Double-quoted strings can be used if there is a compelling escape or formatting reason for using single quotes (e.g. a single quote appears inside the string).

* Add convenience imports in `__init__.py` of a package for shallow first-level repetitive imports, but not for subpackages, even if that subpackage is defined in a single `.py` file.

    For instance, if an import line reads `from garage.foo.bar import Bar` then you should add `from garage.foo.bar import Bar` to `garage/foo/__init__.py` so that users may instead write `from garage.foo import Bar`. However, if an import line reads `from garage.foo.bar.stuff import Baz`, *do not* add `from garage.foo.bar.stuff import Baz` to `garage/foo/__init__.py`, because that obscures the `stuff` subpackage.

    *Do*

    `garage/foo/__init__.py`:
    ```python
    """Foo package."""
    from garage.foo.bar import Bar
    ```
    `garage/barp/bux.py`:
    ```python
    """Bux tools for barps."""
    from garage.foo import Bar
    from garage.foo.stuff import Baz
    ```

    *Don't*

    `garage/foo/__init__.py`:
    ```python
    """Foo package."""
    from garage.foo.bar import Bar
    from garage.foo.bar.stuff import Baz
    ```
    `garage/barp/bux.py`:
    ```python
    """Bux tools for barps."""
    from garage.foo import Bar
    from garage.foo import Baz
    ```
* Imports within the same package should be absolute, to avoid creating circular dependencies due to convenience imports in `__init__.py`

    *Do*

    `garage/foo/bar.py`
    ```python
    from garage.foo.baz import Baz

    b = Baz()
    ```

    *Don't*

    `garage/foo/bar.py`
    ```python
    from garage.foo import Baz  # this could lead to a circular import, if Baz is imported in garage/foo/__init__.py

    b = Baz()
    ```

* Base and interface classes (i.e. classes which are not intended to ever be instantiated) should use the `abc` package to declare themselves as abstract.

   i.e. your class should inherit from `abc.ABC` or use the metaclass `abc.ABCMeta`, it should declare its methods abstract (e.g. using `@abc.abstractmethod`) as-appropriate. Abstract methods should all use `pass` as their implementation, not `raise NotImplementedError`

   *Do*
   ```python
   import abc

   class Robot(abc.ABC):
       """Interface for robots."""

       @abc.abstractmethod
       def beep(self):
           pass
    ```

    *Don't*
    ```python

    class Robot(object):
        "Base class for robots."""

        def beep(self):
            raise NotImplementedError
    ```

* When using external dependencies, use the `import` statement only to import whole modules, not individual classes or functions.

    This applies to both packages from the standard library and 3rd-party dependencies. If a package has a long or cumbersome full path, or is used very frequently (e.g. `numpy`, `tensorflow`), you may use the keyword `as` to create a file-specific name which makes sense. Additionally, you should always follow the community concensus short names for common dependencies (see below).

    *Do*
    ```python
    import collections

    import gym.spaces

    from garage.tf.models import MLPModel

    q = collections.deque(10)
    d = gym.spaces.Discrete(5)
    m = MLPModel(output_dim=2)
    ```

    *Don't*
    ```python
    from collections import deque

    from gym.spaces import Discrete
    import tensorflow as tf

    from garage.tf.models import MLPModel

    q = deque(10)
    d = Discrete(5)
    m = MLPModel(output_dim=2)
    ```

    *Known community-concensus imports*
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_probability as tfp
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import dowel.logger as logger
    import dowel.tabular as tabular
    ```

## Documentation
Python files should provide docstrings for all public methods which follow [PEP257](https://www.python.org/dev/peps/pep-0257/) docstring conventions and [Google](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring formatting. A good docstring example can be found [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Additional standards:
* Docstrings for `__init__` should be included in the class docstring as suggested in the [Google example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* Docstrings should provide full type information for all arguments, return values, exceptions, etc. according to the Google format

### Application guide
**Newly created** Python files should follow all of the above standards for docstrings.

**Non-trivially modified** Python files should be submitted with updated docstrings according to the above standard.

**New or heavily-redesigned** modules with non-trivial APIs and functionality should provide full text documentation, in addition to docstrings, which covers:
* Explanation of the purpose of the module or API
* Brief overview of its design
* Usage examples for the most common use cases
* Explicitly calls out common gotchas, misunderstandings, etc.
* A quick summary of how to go about advanced usage, configuration, or extension

### Other languages
Non-Python files (including XML, HTML, CSS, JS, and Shell Scripts) should follow the [Google Style Guide](https://github.com/google/styleguide) for that language

YAML files should use 2 spaces for indentation.

### Whitespace (all languages)
* Use Unix-style line endings
* Trim trailing whitespace from all lines
* All files should end in a single newline

## Testing
garage maintains a test suite to ensure that future changes do not break existing functionality. We use TravisCI to run a unit test suite on every pull request before merging.

* New functionality should always include unit tests and, where appropriate, integration tests.
* PRs fixing bugs which were not caught by an existing test should always include a test replicating the bug

### Creating Tests
Add a test for your functionality under the `garage/tests/` directory. Make sure your test filename is prepended with test(i.e. `test_<filename>.py`) to ensure the test will be run in the CI.

## Git

### Workflow
__garage uses a linear commit history and rebase-only merging.__

This means that no merge commits appear in the project history. All pull requests, regardless of number of commits, are squashed to a single atomic commit at merge time.

Do's and Don'ts for avoiding accidental merge commits and other headaches:
* *Don't* use GitHub's "Update branch" button on pull requests, no matter how tempting it seems
* *Don't* use `git merge`
* *Don't* use `git pull` (unless git tells you that your branch can be fast-forwarded)
* *Don't* make commits in the `master` branch---always use a feature branch
* *Do* fetch upstream (`rlworkgroup/garage`) frequently and keep your `master` branch up-to-date with upstream
* *Do* rebase your feature branch on `master` frequently
* *Do* keep only one or a few commits in your feature branch, and use `git commit --amend` to update your changes. This helps prevent long chains of identical merges during a rebase.

Please see [this guide](https://gist.github.com/markreid/12e7c2203916b93d23c27a263f6091a0) for a tutorial on the workflow. Note: unlike the guide, we don't use separate `develop`/`master` branches, so all PRs should be based on `master` rather than `develop`

### Commit message format
garage follows the git commit message guidelines documented [here](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53) and [here](https://chris.beams.io/posts/git-commit/). You can also find an in-depth guide to writing great commit messages [here](https://github.com/RomuloOliveira/commit-messages-guide/blob/master/README.md)

In short:
* All commit messages have an informative subject line of 50 characters
* A newline between the subject and the body
* If relevant, an informative body which is wrapped to 72 characters

### Git recipes

These recipes assume you are working out of a private GitHub fork.

If you are working directly as a contributor to `rlworkgroup`, you can replace references to `rlworkgroup` with `origin`. You also, of course, do not need to add `rlworkgroup` as a remote, since it will be `origin` in your repository.

#### Clone your GitHub fork and setup the rlworkgroup remote
```sh
git clone git@github.com:<your_github_username>/garage.git
cd garage
git remote add rlworkgroup git@github.com:rlworkgroup/garage.git
git fetch rlworkgroup
```

#### Update your GitHub fork with the latest from upstream
```sh
git fetch rlworkgroup
git reset --hard master rlworkgroup/master
git push -f origin master
```

#### Make a new feature branch and push it to your fork
```sh
git checkout master
git checkout -b myfeaturebranch
# make some changes
git add file1 file2 file3
git commit # Write a commit message conforming to the guidelines
git push origin myfeaturebranch
```

#### Rebase a feature branch so it's up-to-date with upstream and push it to your fork
```sh
git checkout master
git fetch rlworkgroup
git reset --hard rlworkgroup/master
git checkout myfeaturebranch
git rebase master
# you may need to manually reconcile merge conflicts here. Follow git's instructions.
git push -f origin myfeaturebranch # -f is frequently necessary because rebases rewrite history
```

## Release

### Modify CHANGELOG.md
For each release in garage, modify [CHANGELOG.md](https://github.com/rlworkgroup/garage/blob/master/CHANGELOG.md) with the most relevant changes from the latest release. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), which adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

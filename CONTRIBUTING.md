# Contributing to garage
We welcome all contributions to garage.

Use this guide to prepare your contribution.

# Post-setup instructions
After activating your conda environment (e.g. `conda activate garage`), make sure you run `pre-commit install` to install the pre-commit hooks to your repository.
This will help streamline the pull request process by catching basic problems locally before they are checked by the CI.

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

## Code style
The Python code in garage conforms to the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard. Please read and understand it in detail.

### garage-specific Python style
These are garage-specific rules which are not part of the aforementioned style guides.
* Python package imports should be sorted alphabetically within their PEP8 groupings. The sorting is alphabetical from left to right, ignoring case and Python keywords (i.e. `import`, `from`, `as`). Notable exceptions apply in `__init__.py` files, where sometimes this rule will trigger a circular import.
* We prefer double-quoted strings (`"foo"`) over single-quoted strings (`'foo'`), unless there is a compelling escape or formatting reason for using single quotes
* Add convenience imports in `__init__.py` of a package for shallow first-level repetitive imports, but not for subpackages, even if that subpackage is defined in a single `.py` file. For instance, if an import line reads `from garage.foo.bar import Bar` then you should add `from garage.foo.bar import Bar` to `garage/foo/__init__.py` so that users may instead write `from garage.foo import Bar`. However, if an import line reads `from garage.foo.bar.stuff import Baz`, *do not* add `from garage.foo.bar.stuff import Baz` to `garage/foo/__init__.py`, because that obscures the `stuff` subpackage.
* Newly created python files should follow [PEP257](https://www.python.org/dev/peps/pep-0257/) docstring conventions and [PEP287](https://www.python.org/dev/peps/pep-0287/) docstring formatting. A good PEP257 example can be found [here](https://blog.dolphm.com/pep257-good-python-docstrings-by-example/).

### Other languages
Non-Python files (including XML, HTML, CSS, JS, and Shell Scripts) should follow the [Google Style Guide](https://github.com/google/styleguide) for that language

YAML files should use 2 spaces for indentation.

### Whitespace (all languages)
* Use Unix-style line endings
* Trim trailing whitespace from all lines
* All files should end in a single newline

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
garage follows the git commit message guidelines documented [here](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53) and [here](https://chris.beams.io/posts/git-commit/).

In short:
* All commit messages have an informative subject line of 50 characters
* A newline between the subject and the body
* If relevant, an informative body which is wrapped to 72 characters

garage currently uses [pre-commit](https://pre-commit.com/#usage), a multi-language package manager for pre-commit hooks, and its out-of-the-box [hooks](https://github.com/pre-commit/pre-commit-hooks).
To setup pre-commit on your local machine:
* `pre-commit install` to install pre-commit into your .git/hooks/ directory
* `pre-commit autoupdate` to update the hooks to the latest version

To run pre-commit:
* `pre-commit run --files *file_name*` to run pre-commit on the specified file.
* `pre-commit run --files *file_name1* *file_name2*` to run on multiple files.

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

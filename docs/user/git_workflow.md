# Git Workflow

This doc describes how to effectively use git for garage development.

## Create a Fork of Garage

If you're not a garage contributor, you should create a GitHub fork of the garage repo and push your changes there. You can later use these changes to submit a pull request and merge them into the garage master branch.

To create a fork, simply navigate to the garage repo [here](https://github.com/rlworkgroup/garage) and click on "fork" in the top right corner. This will create clone garage into your own GitHub repo, which you can then make changes to.

## Clone Your Fork Locally

After creating a fork, you'll want to `git clone` it so that it is available on your local machine.  `cd` into your desired parent directory, then:

```sh
git clone git@github.com:<your_github_username>/garage.git
cd garage
```

## Create a Feature Branch

Once you have a local copy, you should create a feature branch off of `master` before implementing a new feature. Do this for every new feature you create - avoid making changes to master, and avoid adding multiple features in one branch unless they are codependent. The `master` branch should only be updated to include the latest changes from the upstream master branch.

You can create a feature branch off master like this:

```sh
git checkout master # make sure you're on the master branch
git checkout -b your-feature-branch # create a new branch called "your-feature-branch"
```

## Passing the Pre-commit Hooks

When making commits, you will need to pass the pre-commit hooks. These hooks run automatically every time you make a commit, assuming you have them installed (if you don't, you can find the installation instructions [in our CONTRIBUTING.md](https://github.com/rlworkgroup/garage/blob/master/CONTRIBUTING.md#preparing-your-repo-to-make-contributions)). They run various check to ensure your changes abide by garage's formatting guidelines and don't introduce unnecessary code, such as  unused import statements or variables.

Note that, in some cases, you may touch a file with code that has not been updated to adhere to garage's latest commit hooks. Garage's policy is that all pre-commit hooks should pass, even if errors are due to existing code not committed by you. It is therefore your responsibility to update the touched files and make them compliant. Exceptions to this rule are only made for large PRs that touch many files.

## Rebase - Don't Merge

If you have implemented a feature, its likely that the changes you introduced were not created on top of the most recent commit on the garage master branch (ie. commits were made to master that you don't have locally). To update your feature branch to include the latest commits made to `rlworkgroup/master`, you should rebase your commits on top of them. This will also avoid conflicts between your commits and the commits on the remote repository.

If you're unfamiliar with rebasing, [this](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase#:~:text=Rebase%20is%20one%20of%20two,has%20powerful%20history%20rewriting%20features.) tutorial explains it in detail. Note that you should *only* rebase when creating a PR, never`git merge`. Garage maintains a linear commit history and does not allow merge commits.

The [CONTRIBUTING.md](https://github.com/rlworkgroup/garage/blob/master/CONTRIBUTING.md#git-recipes) includes instructions on how to rebase, but they are reiterated here for further clarity:

 **1. Make sure you have the rlworkgroup repo as a remote:**

```sh
git remote add rlworkgroup git@github.com:rlworkgroup/garage.git
```

**2. Update your local master branch to include the latest changes to the
upstream repo**

Note that doing this will overrwrite any changes you've made to your master branch locally. Make sure you move any changes you've made to a new branch first, then:

```sh
git checkout master
git fetch rlworkgroup
git reset --hard rlworkgroup/master # update the local master branch
git push -f origin master # optional, this updates the master branch on your fork
```

**3. Rebase your changes onto master:**

```sh
git checkout your-feature-branch
git rebase master # you may have to resolve conflicts here. Follow git's instructions on how to do this.
```

You can verify that the rebase was successful by doing another rebase with `git rebase master`. You should see a message telling you the current branch is up to date.

## Create a PR (Optional)

If you feel that your changes would be helpful to others and want to make available to all garage users, you should submit a pull request. [This](preparing_a_pr.md) doc describes the process of creating a PR and merging it into the garage master branch.

### Backporting a Fix

Backporting is the process of applying a change or bug fix to previous releases that are still supported. Generally speaking, bug fixes should be backported, unleses they require major changes to the release. To backport a change, submit a PR just as you would for merging your change to the master branch, except specify the desired release as the target. See [this](preparing_a_pr.md) doc for more details.

----

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

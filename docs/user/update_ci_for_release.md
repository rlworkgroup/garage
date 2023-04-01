# CI

The CI must be updated whenever there is a new release

### How to create new CI for release

#### Make new scheduled CI for release branch
**1. create new CI file in `.github/workflows` named `ci-<release_version>` in the master branch**

**2. copy the workflow in the `ci-release-TEMPLATE` file over to the `ci-<release_version>` file**

**3. rename the `name` attribute at the top of the file to `CI <release_version>`**

**4. Fill in the checkout action placeholder, use `with: release-<branch-name>` for checkout action**

**5. Uncomment the `schedule` trigger. It's also fine to delete the `pull-request` trigger, that will be handled in the next step**

###backport CI to release branch**

**1. create new CI file in `.github/workflows` named `ci-<release_version>` in the master branch**

**2. copy the workflow in the `ci-release-TEMPLATE` file over to the `ci-<release_version>` file**

**3. rename the `name` attribute at the top of the file to `CI <release_version>`**

**4. Fill in the checkout action placeholder, use `with: fetch-depth: 0` for checkout action**

**5. When creating a release branch, delete all the branch's CI files and add the new ci file.**

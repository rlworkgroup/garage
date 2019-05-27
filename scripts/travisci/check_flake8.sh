#!/usr/bin/env bash

# Python packages considered local to the project, for the purposes of import
# order checking. Comma-delimited.
garage_packages="tests,garage,sandbox,examples"

### ALL FILES ###

# Error codes ignored for all files
ignored_errors_all=(
D     # docstring rules disabled
W503  # line break before binary operator
W504  # line break after binary operator
)

# Files or directories to exclude from checks applied to all files.
exclude_all=(
./tests/'*'
.git
__pycache__
$(cat .gitignore)
*.md
)


### CHANGED FILES ###

# Error codes ignored for changed files
ignored_errors_changed=(
D     # docstring rules disabled
# We prefer break after binary operator, but YAPF breaks before and the behavior
# is not configurable, so we disable related rules in flake8.
# Ref: https://github.com/google/yapf/issues/647
W503  # line break before binary operator
W504  # line break after binary operator
)

# Files or directories to exclude from checks applied to changed files.
exclude_changed=(
./tests/'*'
$(cat .gitignore)
*.md
)


### ADDED FILES ###

# Error codes applied to added files
ignored_errors_added=(
D107  # missing docstring in __init__
W503  # line break before binary operator
W504  # line break after binary operator
)

# Files or directories to exclude from checks applied to added files.
exclude_added=(
./tests/'*'
$(cat .gitignore)
*.md
)


### ALL TEST FILES ###

# Error codes ignored for all test files
test_ignored_errors_all=(
D     # docstring rules disabled
W503  # line break before binary operator
W504  # line break after binary operator
)

# Files or directories to exclude from checks applied to all files.
test_exclude_all=(
.git
__pycache__
$(cat .gitignore)
*.md
)


### CHANGED TEST FILES ###

# Error codes ignored to changed test files
test_ignored_errors_changed=(
D  # docstring rules disabled
W503  # line break before binary operator
)

# Files or directories to exclude from checks applied to changed test files.
test_exclude_changed=(
$(cat .gitignore)
*.md
)


### ADDED TEST FILES ###

# Error codes ignored to added test files
test_ignored_errors_added=(
D  # docstring rules disabled
W503  # line break before binary operator
)

# Files or directories to exclude from checks applied to added test files.
test_exclude_added=(
$(cat .gitignore)
*.md
)


################################################################################
# If Travis CI is running this script and there's a valid pull request,
# use the commits defined by TRAVIS_COMMIT_RANGE to get a list of changed
# and added files introduced in the feature branch,
# Otherwise, obtain the lists by comparing against the master branch in the
# repository.
if [[ "${TRAVIS}" == "true" && "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  files_changed="$(git diff "${TRAVIS_COMMIT_RANGE}" -M --diff-filter={M,R} \
                     --name-only | grep ".*\.py$")"
  files_added="$(git diff "${TRAVIS_COMMIT_RANGE}" -M --diff-filter=A \
                   --name-only | grep ".*\.py$")"
else
  git remote set-branches --add origin master
  git fetch
  files_changed="$(git diff origin/master -M --diff-filter={M,R} --name-only \
                     | grep ".*\.py$")"
  files_added="$(git diff origin/master -M --diff-filter=A --name-only \
                   | grep ".*\.py$")"
fi

# Obtain the files that have been added or modified in the repository that
# exist inside the tests folder.
test_files_changed="$(echo "${files_changed}" |  grep "tests/*")"
test_files_added="$(echo "${files_added}" |  grep "tests/*")"

# Exit status of this script
status=0

# Check rules with flake8
check_flake8() {
  flake8 --isolated \
         --import-order-style=google \
         --application-import-names="${garage_packages}" \
         --per-file-ignores="./src/garage/misc/krylov.py:N802,N803,N806" \
         "$@"
  status="$((${status} | ${?}))"
}

# All files
ignored_errors_all="${ignored_errors_all[@]}"
exclude_all="${exclude_all[@]}"
check_flake8 --ignore="${ignored_errors_all// /,}" \
             --exclude="${exclude_all// /,}"

# Changed files
ignored_errors_changed="${ignored_errors_changed[@]}"
exclude_changed="${exclude_changed[@]}"
if [[ ! -z "${files_changed}" ]]; then
  check_flake8 --ignore="${ignored_errors_changed// /,}" \
               --exclude="${exclude_changed// /,}" \
               ${files_changed}
fi

# Added files
ignored_errors_added="${ignored_errors_added[@]}"
exclude_added="${exclude_added[@]}"
if [[ ! -z "${files_added}" ]]; then
  check_flake8 --ignore="${ignored_errors_added// /,}" \
               --exclude="${exclude_added// /,}" \
               ${files_added}
fi

# All test files
test_ignored_errors_all="${test_ignored_errors_all[@]}"
test_exclude_all="${test_exclude_all[@]}"
check_flake8 --ignore="${test_ignored_errors_all// /,}" \
             --exclude="${test_exclude_all// /,}" \
             --filename="./tests/*"

# Changed test files
test_ignored_errors_changed="${test_ignored_errors_changed[@]}"
test_exclude_changed="${test_exclude_changed[@]}"
if [[ ! -z "${test_files_changed}" ]]; then
  check_flake8 --ignore="${test_ignored_errors_changed// /,}" \
               --exclude="${test_exclude_changed// /,}" \
               ${test_files_changed}
fi

# Added test files
test_ignored_errors_added="${test_ignored_errors_added[@]}"
test_exclude_added="${test_exclude_added[@]}"
if [[ ! -z "${test_files_added}" ]]; then
  check_flake8 --ignore="${test_ignored_errors_added// /,}" \
               --exclude="${test_exclude_added// /,}" \
               ${test_files_added}
fi

exit "${status}"

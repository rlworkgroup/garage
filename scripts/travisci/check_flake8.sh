#!/usr/bin/env bash

# Python packages considered local to the project, for the purposes of import
# order checking. Comma-delimited.
garage_packages="garage,sandbox,examples,tests"

### ALL FILES ###

# Error codes applied to all files
errors_all=(
# TABS
E101 # indentation contains mixed spaces and tabs
W191 # indentation contains tabs
# WHITESPACE IN EXPRESSION AND STATEMENTS
E201 # whitespace after '(', '[' or '{'
E202 # whitespace before ')', ']' or '}'
E241 # multiple spaces after ','
E211 # whitespace before '(' or '['
E221 # multiple spaces before operator
W291 # trailing whitespace
E702 # multiple statements on one line (semicolon)
# PROGRAMMING RECOMMENDATIONS
E714 # use is not operator rather than not ... is
# Unused imports
F401
)

# Error codes ignored for all files
ignored_errors_all=(
)

# Files or directories to exclude from checks applied to all files.
exclude_all=(
./tests/'*'
)


### CHANGED FILES ###

# Error codes applied to changed files
errors_changed=(
# INDENTATION
E125 # continuation line with same indent as next logical line
E128 # continuation line under-indented for visual indent
# MAXIMUM LINE LENGTH
E501 # line too long
# BLANK LINE
E301 # expected 1 blank line, found 0
E302 # expected 2 blank lines, found 0
E303 # too many blank lines (2)
# IMPORTS
E401 # multiple imports on one line
I100 # import statements are in the wrong order
I101 # the names in from import are in the wrong order
I201 # missing newline between import groups
I202 # additional newline in a group of imports
# WHITESPACE IN EXPRESSION AND STATEMENTS
E203 # whitespace before ':', ';' or ','
E225 # missing whitespace around operator
E226 # missing whitespace around arithmetic operator
E251 # unexpected spaces around keyword / parameter equals
E231 # missing whitespace after ':'
E701 # multiple statements on one line (colon)
# COMMENTS
E261 # at least two spaces before inline comment
E262 # inline comment should start with '# '
# NAMING CONVENTIONS
# Names to avoid
E741 # do not use variables named ‘l’, ‘O’, or ‘I’
E742 # do not define classes named ‘l’, ‘O’, or ‘I’
E743 # do not define functions named ‘l’, ‘O’, or ‘I’
# The following error code enables all the error codes for naming conventions
# defined here:
# https://github.com/PyCQA/pep8-naming
# If one of the errors needs to be ignored, just add it to the ignore array.
N
# PROGRAMMING RECOMMENDATIONS
E711 # comparisons to None should always be done with is or is not, never the
     # equality operators
E712 # comparison to True should be 'if cond is True:' or 'if cond:'
E721 # do not compare types, use 'isinstance()'
E722 # do not use bare except, specify exception instead
E731 # do not assign a lambda expression, use a def
)

# Error codes ignored for changed files
ignored_errors_changed=(
# BREAK BEFORE BINARY OPERATOR
# It enforces the break after the operator, which is acceptable, but it's
# preferred to do it before the operator. Since YAPF enforces the preferred
# style, this rule is ignored.
W503 # line break before binary operator
)

# Files or directories to exclude from checks applied to changed files.
exclude_changed=(
./tests/'*'
)

### ADDED FILES ###

# Error codes for added files
errors_added=(
${errors_changed[@]}
# DOCSTRING
# The following error code enables all the error codes for docstring defined
# here:
# http://pep257.readthedocs.io/en/latest/error_codes.html
# If one of the errors needs to be ignored, just add it to the ignore array.
D
)

# Error codes applied to added files
ignored_errors_added=(
)

# Files or directories to exclude from checks applied to added files.
exclude_added=(
./tests/'*'
)

### ALL TEST FILES ###

# Error codes applied to all test files
test_errors_all=(
${errors_all[@]}
)

# Error codes ignored for all test files
test_ignored_errors_all=(
)

# Files or directories to exclude from checks applied to all files.
test_exclude_all=(
)

### CHANGED TEST FILES ###

# Error codes applied to changed test files
test_errors_changed=(
${errors_changed[@]}
)

# Error codes ignored to changed test files
test_ignored_errors_changed=(
)

# Files or directories to exclude from checks applied to changed test files.
test_exclude_changed=(
)

### ADDED TEST FILES ###

# Error codes applied to added test files
test_errors_added=(
${errors_changed[@]}
)

# Error codes ignored to added test files
test_ignored_errors_added=(
)

# Files or directories to exclude from checks applied to added test files.
test_exclude_added=(
)


################################################################################
# If Travis CI is running this script and there's a valid pull request,
# use the commits defined by TRAVIS_COMMIT_RANGE to get a list of changed
# and added files introduced in the feature branch,
# Otherwise, obtain the lists by comparing against the master branch in the
# repository.
if [[ "${TRAVIS}" == "true" && "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  files_changed="$(git diff "${TRAVIS_COMMIT_RANGE}" -M --diff-filter={M,R} \
                     --name-only | grep ".*\.py")"
  files_added="$(git diff "${TRAVIS_COMMIT_RANGE}" -M --diff-filter=A \
                   --name-only | grep ".*\.py")"
else
  git remote set-branches --add origin master
  git fetch
  files_changed="$(git diff origin/master -M --diff-filter={M,R} --name-only \
                     | grep ".*\.py")"
  files_added="$(git diff origin/master -M --diff-filter=A --name-only \
                   | grep ".*\.py")"
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
         "$@"
  status="$((${status} | ${?}))"
}

# All files
errors_all="${errors_all[@]}"
ignored_errors_all="${ignored_errors_all[@]}"
exclude_all="${exclude_all[@]}"
check_flake8 --select="${errors_all// /,}" \
             --ignore="${ignored_errors_all// /,}" \
             --exclude="${exclude_all// /,}"

# Changed files
errors_changed="${errors_changed[@]}"
ignored_errors_changed="${ignored_errors_changed[@]}"
exclude_changed="${exclude_changed[@]}"
if [[ ! -z "${files_changed}" ]]; then
  check_flake8 --select="${errors_changed// /,}" \
               --ignore="${ignored_errors_changed// /,}" \
               --exclude="${exclude_changed// /,}" \
               ${files_changed}
fi

# Added files
errors_added="${errors_added[@]}"
ignored_errors_added="${ignored_errors_added[@]}"
exclude_added="${exclude_added[@]}"
if [[ ! -z "${files_added}" ]]; then
  check_flake8 --select="${errors_added// /,}" \
               --ignore="${ignored_errors_added// /,}" \
               --exclude="${exclude_added// /,}" \
               ${files_added}
fi

# All test files
test_errors_all="${test_errors_all[@]}"
test_ignored_errors_all="${test_ignored_errors_all[@]}"
test_exclude_all="${test_exclude_all[@]}"
check_flake8 --select="${test_errors_all// /,}" \
             --ignore="${test_ignored_errors_all// /,}" \
             --exclude="${test_exclude_all// /,}" \
             --filename="./tests/*"

# Changed test files
test_errors_changed="${test_errors_changed[@]}"
test_ignored_errors_changed="${test_ignored_errors_changed[@]}"
test_exclude_changed="${test_exclude_changed[@]}"
if [[ ! -z "${test_files_changed}" ]]; then
  check_flake8 --select="${test_errors_changed// /,}" \
               --ignore="${test_ignored_errors_changed// /,}" \
               --exclude="${test_exclude_changed// /,}" \
               ${test_files_changed}
fi

# Added test files
test_errors_added="${test_errors_added[@]}"
test_ignored_errors_added="${test_ignored_errors_added[@]}"
test_exclude_added="${test_exclude_added[@]}"
if [[ ! -z "${test_files_added}" ]]; then
  check_flake8 --select="${test_errors_added// /,}" \
               --ignore="${test_ignored_errors_added// /,}" \
               --exclude="${test_exclude_added// /,}" \
               ${test_files_added}
fi

exit "${status}"

#!/usr/bin/env bash

# Python packages considered local to the project, for the purposes of import
# order checking. Comma-delimited.
garage_packages="garage,sandbox,examples,contrib"

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
)

# Error codes ignored for all files
ignored_errors_all=(
)

#errors_all="${errors_all[@]}"
#ignored_errors_all="${ignored_errors_all[@]}"
#flake8 --isolated --select="${errors_all// /,}" \
#  --ignore="${ignored_errors_all// /,}"


### CHANGED_FILES ###

# Error codes applied to changed files
errors_changed=(
# INDENTATION
E125 # continuation line with same indent as next logical line
E128 # continuation line under-indented for visual indent
# MAXIMUM LINE LENGTH
E501 # line too long
# BREAK BEFORE BINARY OPERATOR
W503 # line break before binary operator
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

# Add the codes of the errors to be ignored for the absolute verification in
# this array.
ignored_errors_changed=(
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

################################################################################

# DOCSTRING
# The following error code enables all the error codes for docstring defined
# here:
# http://pep257.readthedocs.io/en/latest/error_codes.html
# If one of the errors needs to be ignored, just add it to the ignore array.
#erros_add="${errors_changed[@]} D"

if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  files_changed="$(git diff "${TRAVIS_COMMIT_RANGE}" --diff-filter=M \
                     --name-only | grep ".*\.py")"
  files_added="$(git diff "${TRAVIS_COMMIT_RANGE}" --diff-filter=A \
                   --name-only | grep ".*\.py")"
else
  git remote set-branches --add origin master
  git fetch
  files_changed="$(git diff origin/master --diff-filter=M --name-only \
                     | grep ".*\.py")"
  files_added="$(git diff origin/master --diff-filter=A --name-only \
                   | grep ".*\.py")"
fi

# Check rules with flake8
check_flake8() {
  flake8 --isolated \
         --import-order-style=google \
         --application-import-names="${garage_packages}" \
         "$@"
}

# All files
errors_all="${errors_all[@]}"
ignored_errors_all="${ignored_errors_all[@]}"
check_flake8 --select="${errors_all// /,}" \
             --ignore="${ignored_errors_all// /,}"

# Changed files
errors_changed="${errors_changed[@]}"
ignored_errors_changed="${ignored_errors_changed[@]}"
if [[ ! -z "${files_changed}" ]]; then
  check_flake8 --select="${errors_changed// /,}" \
               --ignore="${ignored_errors_changed// /,}" ${files_changed}
fi

# Added files
errors_added="${errors_added[@]}"
ignored_errors_added="${ignored_errors_added[@]}"
if [[ ! -z "${files_added}" ]]; then
  check_flake8 --select="${errors_added// /,}" \
               --ignore="${ignored_errors_added// /,}" ${files_added}
fi

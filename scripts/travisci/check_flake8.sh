#!/usr/bin/env bash
# ABSOLUTE VERIFICATION:
# The following errors are analyzed in all python files across the repository.
errors_absolute=(
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

# Add the codes of the errors to be ignored for the absolute verification in
# this array.
ignored_errors_absolute=(
)

errors_absolute="${errors_absolute[@]}"
ignored_errors_absolute="${ignored_errors_absolute[@]}"
flake8 --isolated --select="${errors_absolute// /,}" \
  --ignore="${ignored_errors_absolute// /,}"


# INCREMENTAL VERIFICATION:
# The following errors are analyzed only in the modified code in the pull
# request branch.
errors_change=(
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
ignored_errors_change=(
)

# DOCSTRING
# The following error code enables all the error codes for docstring defined
# here:
# http://pep257.readthedocs.io/en/latest/error_codes.html
# If one of the errors needs to be ignored, just add it to the ignore array.
erros_add="${errors_change[@]} D"

errors_change="${errors_change[@]}"
ignored_errors_change="${ignored_errors_change[@]}"
if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  files_changed="$(git diff "${TRAVIS_COMMIT_RANGE}" --diff-filter=M \
                     --name-only | grep ".*.py")"
  files_added="$(git diff "${TRAVIS_COMMIT_RANGE}" --diff-filter=A \
                   --name-only | grep ".*.py")"
  flake8 --isolated --import-order-style=google \
    --application-import-names=sandbox,garage,examples,contrib \
    --select="${errors_change// /,}" \
    --ignore="${ignored_errors_change// /,}" ${files_changed}
  flake8 --isolated --import-order-style=google \
    --application-import-names=sandbox,garage,examples,contrib \
    --select="${erros_add// /,}" \
    --ignore="${ignored_errors_change// /,}" ${files_added}
else
  git remote set-branches --add origin master
  git fetch
  files_changed="$(git diff origin/master --diff-filter=M --name-only \
                     | grep ".*.py")"
  files_added="$(git diff origin/master --diff-filter=A --name-only \
                   | grep ".*.py")"
  flake8 --isolated --import-order-style=google \
    --application-import-names=sandbox,garage,examples,contrib \
    --select="${errors_change// /,}" \
    --ignore="${ignored_errors_change// /,}" ${files_changed}
  flake8 --isolated --import-order-style=google \
    --application-import-names=sandbox,garage,examples,contrib \
    --select="${erros_add// /,}" \
    --ignore="${ignored_errors_change// /,}" ${files_added}
fi

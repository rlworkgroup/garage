#!/usr/bin/env bash
# Exit status of this script
exit_status=0
# Absolute verification:
# The following errors are analyzed in all python files across the repository.
#
# TABS
# W191 - Indentation contains tabs
# E101 - Indentation contains mixed spaces and tabs
errors="E101,W191,"

# WHITESPACE IN EXPRESSION AND STATEMENTS
# E201 whitespace after '(', '[' or '{'
# E202 whitespace before ')', ']' or '}'
errors+="E201,E202,"
# E241 multiple spaces after ','
errors+="E241,"
# E211 whitespace before '(' or '['
errors+="E211"
# E221 multiple spaces before operator
errors+="E221,"
# W291 trailing whitespace
errors+="W291"
# E702 multiple statements on one line (semicolon)
errors+="E702"
find -iname "*.py" | flake8 --select=$errors
if [[ $? -ne 0 ]]; then
  exit_status=1
fi


# INCREMENTAL VERIFICATION:
# The following errors are analyzed only in the modified code in the pull
# request branch.
#
# INDENTATION
# E125 continuation line with same indent as next logical line
# E128 continuation line under-indented for visual indent
errors="E125,E128,"

# MAXIMUM LINE LENGTH
# E501 line too long
errors+="E501,"

# BREAK BEFORE BINARY OPERATOR
# W503 line break before binary operator
errors+="W503,"

# BLANK LINE
# E301 expected 1 blank line, found 0
# E302 expected 2 blank lines, found 0
# E303 too many blank lines (2)
errors+="E301,E302,E303,"

# IMPORTS
# E100 Import statements are in the wrong order
# E401 multiple imports on one line
# I201 Missing newline between import groups
errors+="E100,E401,I201,"

# WHITESPACE IN EXPRESSION AND STATEMENTS
# E203 whitespace before ':', ';' or ','
errors+="E203,"
# E225 missing whitespace around operator
# E226 missing whitespace around arithmetic operator
errors+="E225,E226,"
# E251 unexpected spaces around keyword / parameter equals
errors+="E251,"
# E231 missing whitespace after ':'
errors+="E231,"
# E701 multiple statements on one line (colon)
errors+="E701,"

# STRING QUOTES
# Use double quote characters to be consistent with the docstring convention
# in PEP 257.
quotes="--docstring-quotes 'double'"
# We prefer double-quoted strings (`"foo"`) over single-quoted strings
# (`'foo'`), unless there is a compelling escape or formatting reason for using
# single quotes. Therefore, ignore those rules to avoid conflicts.
# Q000 Remove bad quotes
# Q001 Remove bad quotes from multiline string
ignore+="Q000,Q001"

if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  git diff ${TRAVIS_COMMIT_RANGE} | flake8 --diff $quotes --ignore=$ignore --select=$errors
  if [[ $? -ne 0 ]]; then
    exit_status=1
  fi
else
  git remote set-branches --add origin master
  git fetch
  git diff origin/master | flake8 --diff $quotes --ignore=$ignore --select=$errors
  if [[ $? -ne 0 ]]; then
    exit_status=1
  fi
fi

exit ${exit_status}

#!/usr/bin/env bash
# Exit status of this script
exit_status=0

# Add an option to run the incremental verification locally to reduce the time
# spent in travis for error verification. It's required to rebase the feature
# branch with the master branch to avoid checking on unknown changes.
while getopts ":l" opt; do
  case ${opt} in
    l)
      TRAVIS_PULL_REQUEST="false"
      ;;
    \?)
      echo "usage: check_flake8 [-l]"
      echo "    -l    run incremental verification locally"
      exit 1
      ;;
  esac
done

# ABSOLUTE VERIFICATION:
# The following errors are analyzed in all python files across the repository.
#
# TABS
# W191 indentation contains tabs
# E101 indentation contains mixed spaces and tabs
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

# PROGRAMMING RECOMMENDATIONS
# E714 use is not operator rather than not ... is.
errors+="E714"

find -iname "*.py" | flake8 --select=${errors}
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
# E100 import statements are in the wrong order
# E401 multiple imports on one line
# I201 missing newline between import groups
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

# COMMENTS
# E261 at least two spaces before inline comment
errors+="E261,"
# E262 inline comment should start with '# '
errors+="E262,"

# NAMING CONVENTIONS
# Names to avoid
# E741 do not use variables named ‘l’, ‘O’, or ‘I’
# E742 do not define classes named ‘l’, ‘O’, or ‘I’
# E743 do not define functions named ‘l’, ‘O’, or ‘I’
errors+="E741,E742,E743,"
# The following error code enables all the error codes for naming conventions
# defined here:
# https://github.com/PyCQA/pep8-naming
# If one of the errors needs to be ignored, just add it to the ignore variable.
errors+="N,"

# DOCSTRING
# The following error code enables all the error codes for docstring defined
# here:
# http://pep257.readthedocs.io/en/latest/error_codes.html
# If one of the errors needs to be ignored, just add it to the ignore variable.
errors+="D,"

# PROGRAMMING RECOMMENDATIONS
# E711 comparisons to None should always be done with is or is not, never the
# equality operators.
errors+="E711,"
# E714 use is not operator rather than not ... is.
errors+="E714,"
# E731 do not assign a lambda expression, use a def.
errors+="E731,"
# E722 do not use bare except, specify exception instead.
errors+="E722"

if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  git diff ${TRAVIS_COMMIT_RANGE} | flake8 --diff ${quotes} --select=${errors} --ignore=${ignore}
  if [[ $? -ne 0 ]]; then
    exit_status=1
  fi
else
  git remote set-branches --add origin master
  git fetch
  git diff origin/master | flake8 --diff ${quotes} --select=${errors} --ignore=${ignore}
  if [[ $? -ne 0 ]]; then
    exit_status=1
  fi
fi

exit ${exit_status}

exit_status=0
# Absolute verification
# W191 - Indentation contains tabs
# E101 - Indentation contains mixed spaces and tabs
find -iname "*.py" | flake8 --select=E101,W191
if [ $? -ne 0 ]; then
  exit_status=1
fi

# Incremental verification
# E501 - Line too long
if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  git diff ${TRAVIS_COMMIT_RANGE} | flake8 --diff --select=E501
  if [ $? -ne 0 ]; then
    exit_status=1
  fi
fi

exit $exit_status

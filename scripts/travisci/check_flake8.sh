# Absolute verification
# W191 - Indentation contains tabs
# E101 - Indentation contains mixed spaces and tabs
find -iname "*.py" | flake8 --select=E101,W191
#flake8 --select=E101,W191 $(git ls-tree --full-tree --name-only -r HEAD | grep \.py)

# Incremental verification
# E501 - Line too long
if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  echo "Commit range: "
  echo ${TRAVIS_COMMIT_RANGE}
  git diff ${TRAVIS_COMMIT_RANGE} | flake8 --diff --select=E501
fi

#!/usr/bin/env bash
status=0

echo "Checking commit range ${TRAVIS_COMMIT_RANGE}"
SOURCE="${TRAVIS_COMMIT_RANGE%...*}"
ORIGIN="${TRAVIS_COMMIT_RANGE#*...}"
pre-commit run --source "${SOURCE}" --origin "${ORIGIN}"
status="$((${status} | ${?}))"

while read commit; do
  echo "Checking commit message for ${commit}"
  commit_msg="$(mktemp)"
  git log --format=%B -n 1 "${commit}" > "${commit_msg}"
  scripts/check_commit_message "${commit_msg}"
  pass=$?
  status="$((${status} | ${pass}))"

  # Print message if it fails
  if [[ "${pass}" -ne 0 ]]; then
    echo "Failing commit message:"
    cat "${commit_msg}"
  fi

done < <(git log --cherry-pick --left-only --pretty="%H" \
                 "${ORIGIN}...${SOURCE}")

exit "${status}"

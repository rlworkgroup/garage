#!/usr/bin/env bash
status=0

echo "Checking commit range ${TRAVIS_COMMIT_RANGE}"
SOURCE="${TRAVIS_COMMIT_RANGE%...*}"
ORIGIN="${TRAVIS_COMMIT_RANGE#*...}"
status="$((${status} | ${?}))"

while read commit; do
  echo "Checking for docs only in ${commit}"
  not_docs="$(git show --name-only --oneline ${commit} \
    | tail -n +2 \
    | awk -F . '{print $NF}' \
    | uniq \
    | grep -v 'md\|rst\|png\|bib\|html\|css')"
  test -z "${not_docs}"
  pass=$?
  status="$((${status} | ${pass}))"

  # Print message if it fails
  if [[ "${pass}" -ne 0 ]]; then
    echo "Found non-documentation changes in commit ${commit}"
  fi

done < <(git log --cherry-pick --left-only --pretty="%H" \
                 "${ORIGIN}...${SOURCE}")

exit "${status}"

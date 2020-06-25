#!/usr/bin/env bash
status=0

function join_by { local d="${1}"; shift; echo -n "${1}"; shift; printf "%s" "${@/#/$d}"; }

deps_files=(
  '^setup.py$'
  '^benchmarks/setup.py$'
  '^Makefile$'
  '^docker/'
)
deps_regex="$(join_by '\|' ${deps_files[@]})"

echo "Checking commit range ${TRAVIS_COMMIT_RANGE}"
SOURCE="${TRAVIS_COMMIT_RANGE%...*}"
ORIGIN="${TRAVIS_COMMIT_RANGE#*...}"
status="$((${status} | ${?}))"

while read commit; do
  echo "Checking for dependency changes in ${commit}"
  deps_change="$(git show --name-only --oneline ${commit} \
    | tail -n +2 \
    | grep "${deps_regex}" \
  )"
  test -z "${deps_change}"
  pass=$?
  status="$((${status} | ${pass}))"

  # Print message if it fails
  if [[ "${pass}" -ne 0 ]]; then
    echo -e "Found dependency changes in ${commit}"
    echo -e "Matched with changes in files:\n${deps_change}"
  fi

done < <(git log --cherry-pick --left-only --pretty="%H" \
                 "${ORIGIN}...${SOURCE}")

exit "${status}"

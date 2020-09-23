#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/utils.sh

# Validate submission
python ${DIR}/validate_config.py
log_success Validated experiment YAML file

print_usage() {
cat << USAGE
Usage: ./utilities/submit.sh "impala-ppo-v0.1"
USAGE
}

check_remote() {
  if ! git remote -v | grep gitlab.aicrowd.com > /dev/null; then
    log_normal AIcrowd remote not found
    log_error Please run \`git remote add aicrowd git@gitlab.aicrowd.com:\<username\>/\<repo\>.git\` and rerun this command
  fi
}

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

submit() {
  check_remote
  REMOTE=$(git remote -v | grep gitlab.aicrowd.com | head -1 | awk '{print $1}')
  TAG=$(echo "$@" | sed 's/ /-/g')
  git add --all
  git commit -m "Changes for submission-$TAG"
  git tag -am "submission-$TAG" "submission-$TAG"
  git push $REMOTE master
  git push $REMOTE "submission-$TAG"
  log_success Visit the issues page of your repository on https://gitlab.aicrowd.com to track the progress of your submission
}

submit "$@"
#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/utils.sh

_teardown_procgen_env() {
  log_info "Removing procgen environment..."
  if which conda > /dev/null 1>&2; then
    . ${HOME}/miniconda3/etc/profile.d/conda.sh 2> /dev/null
  fi
  conda env remove -n procgen
  log_success "Removed procgen environment!"
}

_teardown_procgen_env

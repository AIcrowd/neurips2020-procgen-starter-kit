#!/bin/bash

set -e

MINICONDA_DOWNLOAD_PATH="/tmp/miniconda.sh"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/utils.sh

_download_miniconda() {
  log_info "Downloading Miniconda..."
  miniconda_base_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-"
  case `uname` in
  "Linux")
    miniconda_file="Linux-x86_64.sh"
    ;;
  "Darwin")
    miniconda_file="MacOSX-x86_64.sh"
    ;;
  *)
    log_error "Sorry, we don't support this platform :("
    ;;
  esac
  wget -O "${MINICONDA_DOWNLOAD_PATH}" "${miniconda_base_url}${miniconda_file}"
  log_success "Downloaded Miniconda!"
}

download_miniconda() {
  if [ -f "${MINICONDA_DOWNLOAD_PATH}" ]; then
    return
  fi
  _download_miniconda
}

_install_miniconda() {
  log_info "Installing Miniconda..."
  bash ${MINICONDA_DOWNLOAD_PATH} -b -p ${HOME}/miniconda3
  . ${HOME}/miniconda3/etc/profile.d/conda.sh
  conda init
}

install_miniconda() {
  if which conda 2> /dev/null 1>&2; then
    log_error "Found an existing conda installation!"
    return
  fi
  # if [ -d "${HOME}/miniconda3" || -d "${HOME}/anaconda3" ]; then
  #   log_error "Found an existing conda installation!"
  #   return
  # fi
  download_miniconda
  _install_miniconda
}

_create_procgen_env() {
  log_info "Creating procgen environment..."
  if which conda > /dev/null 1>&2; then
    . ${HOME}/miniconda3/etc/profile.d/conda.sh 2> /dev/null
  fi
  conda create -n procgen -y
  conda activate procgen
  conda install python=3.7 -y
  pip install ray[rllib]==0.8.6 procgen
  log_success "Created procgen environment!"
}

create_procgen_env() {
  if [ -d "${HOME}/miniconda3/envs/procgen" ] || [ -d "${HOME}/anaconda3/envs/procgen" ]; then
    log_error "Found an existing procgen environment"
    log_normal "If this is not expected, please run \`./utilities/teardown.sh\` and re-run \`./utilities/setup.sh\`"
    return
  fi
  _create_procgen_env
  log_normal "You can activate the procgen environment by running \`conda activate procgen\`"
  log_info "###### NOTE ######"
  log_normal "Please install your desired version of PyTorch or Tensorflow."
  log_normal "Note that the maximum supported CUDA version during evaluations is CUDA 10.1"
  log_info "##################"

  exec ${SHELL}
}

setup_procgen_env() {
  install_miniconda
  create_procgen_env
}

setup_procgen_env


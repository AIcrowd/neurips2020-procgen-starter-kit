#!/bin/bash

log_info() {
  echo -e "\033[0;36m$@\033[0m"
}

log_success() {
  echo -e "\033[0;32m$@\033[0m"
}

log_normal() {
  echo -e "$@"
}

log_error() {
  echo -e "\033[0;31m$@\033[0m"
}

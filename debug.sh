#!/bin/bash

export OUTPUTS_DIR=./outputs
rm -rf ${OUTPUTS_DIR}
mkdir ${OUTPUTS_DIR}

./run.sh

#!/bin/bash

export OUTPUTS_DIR=./outputs
export RAY_MEMORY_LIMIT=1500000000
export RAY_CPUS=2
export RAY_STORE_MEMORY=1000000000

rm -rf ${OUTPUTS_DIR}
mkdir ${OUTPUTS_DIR}

./run.sh

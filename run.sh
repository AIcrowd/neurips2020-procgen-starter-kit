#!/bin/bash

python train.py -f experiments/procgen-0.yaml --ray-memory 1500000000 --ray-num-cpus 2 --ray-object-store-memory 1000000000
STATUS_CODE=$?

sleep 100000

mv ray_results/procgen ${OUTPUTS_DIR}/ray
mv mlruns ${OUTPUTS}/mlflow

exit $STATUS_CODE

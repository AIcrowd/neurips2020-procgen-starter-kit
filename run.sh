#!/bin/bash

python train.py -f experiments/procgen-0.yaml --local-dir ${OUTPUTS_DIR}
STATUS_CODE=$?

sleep 100000

mv ray_results/procgen ${OUTPUTS_DIR}/ray
mv mlruns ${OUTPUTS}/mlflow

exit $STATUS_CODE

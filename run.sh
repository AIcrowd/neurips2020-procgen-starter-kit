#!/bin/bash

python train.py -f experiments/procgen-0.yaml --local-dir ${OUTPUTS_DIR}
STATUS_CODE=$?

mv ray_results ${OUTPUTS_DIR}/ray
mv mlflow ${OUTPUTS}/mlflow

exit $STATUS_CODE

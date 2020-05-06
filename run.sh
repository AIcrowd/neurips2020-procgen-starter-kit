#!/bin/bash

python train.py -f experiments/procgen-0.yaml --local-dir ${OUTPUTS_DIR}
STATUS_CODE=$?

ls

ls /tmp

mv /tmp/ray_results/procgen ${OUTPUTS_DIR}/ray
mv mlflow ${OUTPUTS}/mlflow

exit $STATUS_CODE

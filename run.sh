#!/bin/bash

python train.py -f experiments/procgen-0.yaml \
    --ray-memory ${RAY_MEMORY_LIMIT} \
    --ray-num-cpus ${RAY_CPUS} \
    --ray-object-store-memory ${RAY_STORE_MEMORY}

STATUS_CODE=$?

mv ray_results/* ${OUTPUTS_DIR}/ray
mv mlruns ${OUTPUTS_DIR}/mlruns

exit $STATUS_CODE

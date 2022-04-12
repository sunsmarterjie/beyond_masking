#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

GPUS=$1
WORK_DIR=$2
PY_ARGS=${@:5}
PORT=${PORT:-29500}

# echo ${CFG%.*}
# WORK_DIR=${WORK_DIR}$(echo ${CFG%.*} | sed -e "s/configs//g")/

echo $WORK_DIR
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    run_mae_pretraining.py --work_dir $WORK_DIR  ${PY_ARGS}

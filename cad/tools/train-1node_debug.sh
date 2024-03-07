#!/bin/bash
#   --gpus-per-node=1 \
GPUS_NUM=4

srun -p work \
 --gres=gpu:$GPUS_NUM \
   --cpus-per-task=8 \
    --mem=64G cutler/tools/single-node_run_debug.sh $GPUS_NUM

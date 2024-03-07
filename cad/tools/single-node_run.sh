#!/bin/bash
export DETECTRON2_DATASETS=/home/ssaricha/CutLER/datasets
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:12399"
SOCKET_NAME=$(ip r | grep default | awk '{print $5}')
echo "Socket name: $SOCKET_NAME"
export GLOO_SOCKET_IFNAME=$SOCKET_NAME
PYTHON=/home/ssaricha/anaconda3/envs/detectron2/bin/python
$PYTHON -u train_net.py --num-gpus 6 --config-file cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_student_teacher.yaml --num-machines 1 --machine-rank "$SLURM_NODEID" --dist-url "$DIST_URL" "$@"


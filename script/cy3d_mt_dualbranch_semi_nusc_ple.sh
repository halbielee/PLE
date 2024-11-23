#!/usr/bin/env bash

# Dataset: NuScenes
# Ratio: 0.5, 1, 2, 5, 10, 20, 50%
# Model: MeanTeacher (Cylinder3D + DualBranch)



GPU=0,1,2,3
SEMI_RATIO=0.5
CONFIG_BASE=configs/ple/meanteacher_nuscenes
CONFIG=$CONFIG_BASE/cy3d_mt_dualbranch_semi_nusc_ple_${SEMI_RATIO}.py
DATA_ROOT=dataset/nuscenes_kitti/
WORK_DIR=YOUR/LOG/PATH
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-32000}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --data-root $DATA_ROOT \
    --work-dir $WORK_DIR \
    --launcher pytorch ${@:3}

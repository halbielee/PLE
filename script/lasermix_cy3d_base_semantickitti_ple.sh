#!/usr/bin/env bash

# Dataset: SemanticKITTI
# Ratio: 20%
# Model: LaserMix + DualBranch
# Since applying PLE on 20% or 50% generates labels for almost all unlabeled data,
# We apply LaserMix as a data augmentation method for this experiment

GPU=0,1,2,3
SEMI_RATIO=20
CONFIG_BASE=configs/ple/lasermix_semantickitti
CONFIG=$CONFIG_BASE/lasermix_cy3d_base_semantickitti_ple_$SEMI_RATIO.py
DATA_ROOT=dataset/SemanticKITTI2/dataset/
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

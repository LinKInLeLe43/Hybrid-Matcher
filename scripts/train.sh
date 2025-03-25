#!/bin/bash -l

conda activate gim

GPUS=8
GITID=$(git rev-parse --short=8 HEAD)
MODELID=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
python train.py \
    --trains MegaDepth ScanNet \
    --batch_size 5 \
    --valid_batch_size 2 \
    --gpus $GPUS \
    --max_epochs 10 \
    --git $GITID \
    --wid $MODELID \
    --img_size 672 \
    --lr 0.001 \
    --min_lr 0.00005 \
    --maxlen 938240 938240 \
    --resample \

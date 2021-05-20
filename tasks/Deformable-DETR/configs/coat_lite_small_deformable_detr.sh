#!/usr/bin/env bash

set -x

EXP_DIR=exps/coat_lite_small_deformable_detr
PY_ARGS=${@:1}

mkdir -p ${EXP_DIR}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone "coat_lite_small" \
    --backbone_weights "../../output/pretrained/coat_lite_small_8d362f48.pth" \
    ${PY_ARGS} | tee -a ${EXP_DIR}/history.txt
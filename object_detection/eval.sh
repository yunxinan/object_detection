#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 ./eval.py \
        --logtostderr \
        --checkpoint_dir=/data/liuyan/identitycard/model/idacrd_model_1 \
        --eval_dir=/data/liuyan/identitycard/model/eval_logs \
        --pipeline_config_path=/home/liuyan/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config \
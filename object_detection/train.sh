#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 ./train.py \
       --logtostderr \
       --pipeline_config_path=/home/liuyan/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config \
       --train_dir=/data/liuyan/identitycard/model/idacrd_model_1/
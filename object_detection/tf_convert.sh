#!/bin/bash

python3 ./create_pascal_tf_record.py \
       --label_map_path=data/pascal_label_map.pbtxt \
       --data_dir=/data/liuyan/drivinglic/imagedata/idcard_1_trainval \
       --year=VOC2007 \
       --set=train \
       --output_path=/data/liuyan/drivinglic/imagedata/tfrecoder/1_object_train/pascal_train.record
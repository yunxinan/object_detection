import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import os
import shutil
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pb_file_path = "/data/liuyan/ocr_model/20180402-114759.pb"
save_path = '/data/liuyan/ocr_model/ckpt/model.ckpt'
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    #    image_path = "/data/wangchongjin/pornographic_recognition/data/porn"
    with sess.as_default():
        v1 = tf.Variable(10, name="v1")
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, save_path)

        print("success")

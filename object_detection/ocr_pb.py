#coding:utf-8
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import shutil
import os.path
import os
import numpy as np
import PIL.Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getImage(path):
    height =30
    width =90
    images_actual_data = np.ndarray(shape=(1, height, width, 3),dtype='uint8')
    pil_image = PIL.Image.open(tf.gfile.GFile(path, 'rb'))
    pil_image = pil_image.resize((width, height), PIL.Image.ANTIALIAS)
    images_actual_data[0, ...] = np.asarray(pil_image)
    print(images_actual_data.shape[0])
    print(images_actual_data.shape[1])
    print(images_actual_data.shape[2])
    print(images_actual_data.shape[3])
    return images_actual_data

imagePath = '/data/loukang/imagedata/idcard_test/idcard_name/ (4).jpg'
images_data = getImage(imagePath)
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = '/data/loukang/idcardmodel/frozen_inference_graph.pb'
    #sess.graph.add_to_collection("input", mnist.test.images)

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:

        tf.initialize_all_variables().run()
        input_x = sess.graph.get_tensor_by_name("Placeholder_1:0")
        print(input_x)
        output = sess.graph.get_tensor_by_name("AttentionOcr_v1/predicted_chars:0")
        print(output)


        predict = sess.run(output,{input_x:images_data})
        print ("predict", predict)

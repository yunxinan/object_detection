#!/usr/bin/python
# encoding: utf-8

import os
import tensorflow as tf
save_path = "/data/liuyan/ocr_model/ckpt/model.ckpt"
#pb_file_path = "porn_model/model.pb"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    saver = tf.train.import_meta_graph("/data/liuyan/ocr_model/ckpt/model.ckpt.meta")
    print("Loading saved model...")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path)
        #x = sess.graph.get_tensor_by_name("data:0")
        input_img = sess.graph.get_tensor_by_name("x:0")
        input_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        output_box = sess.graph.get_tensor_by_name("fc/Relu:0")
        #prob = sess.graph.get_tensor_by_name("prob:0")
        #values, indices = tf.nn.top_k(prob, 3)
        print(input_img)
        print(input_prob)
        print(output_box)

        export_path = '/data/liuyan/ocr_model/tf-wpai-serving-driving-bboxregress'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_input_img = tf.saved_model.utils.build_tensor_info(input_img)
        tensor_info_input_prob = tf.saved_model.utils.build_tensor_info(input_prob)
        tensor_info_boxes = tf.saved_model.utils.build_tensor_info(output_box)
        #tensor_info_pro = tf.saved_model.utils.build_tensor_info(tf.reshape(values, [3]))
        #tensor_info_classify = tf.saved_model.utils.build_tensor_info(tf.reshape(indices,[3]))
        signature_def_map = {
            "drive_bboxregress": tf.saved_model.signature_def_utils.build_signature_def(   #模型导出时签名参数
                 inputs={"image": tensor_info_input_img,  #input  key
                         "prob" : tensor_info_input_prob},  #input  key
                 outputs={
                     "box": tensor_info_boxes,  #output key
                 },
                 method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
             )}
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map=signature_def_map)
        builder.save()
        print('builder.save finished.')

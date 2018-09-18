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
        input_img = sess.graph.get_tensor_by_name("input:0")
        input_phase_train = sess.graph.get_tensor_by_name("phase_train:0")
        output_feature = sess.graph.get_tensor_by_name("embeddings:0")
        #prob = sess.graph.get_tensor_by_name("prob:0")
        #values, indices = tf.nn.top_k(prob, 3)
        print(input_img)
        print(input_phase_train)
        print(output_feature)

        export_path = '/data/liuyan/ocr_model/tf-wpai-serving-face-recognition'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_input_img = tf.saved_model.utils.build_tensor_info(input_img)
        tensor_info_input_prob = tf.saved_model.utils.build_tensor_info(input_phase_train)
        tensor_info_boxes = tf.saved_model.utils.build_tensor_info(output_feature)
        #tensor_info_pro = tf.saved_model.utils.build_tensor_info(tf.reshape(values, [3]))
        #tensor_info_classify = tf.saved_model.utils.build_tensor_info(tf.reshape(indices,[3]))
        signature_def_map = {
            "face_recognition": tf.saved_model.signature_def_utils.build_signature_def(   #模型导出时签名参数
                 inputs={"image": tensor_info_input_img,  #input  key
                         "phase" : tensor_info_input_prob},  #input  key
                 outputs={
                     "embed": tensor_info_boxes,  #output key
                 },
                 method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
             )}
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map=signature_def_map)
        builder.save()
        print('builder.save finished.')

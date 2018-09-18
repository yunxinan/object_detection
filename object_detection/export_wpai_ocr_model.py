
#!/usr/bin/python
# encoding: utf-8

import os
import tensorflow as tf
save_path = "/data/liuyan/ocr_model/pb_to_ckpt_ocrmodel/model.ckpt"
#pb_file_path = "porn_model/model.pb"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    saver = tf.train.import_meta_graph("/data/liuyan/ocr_model/pb_to_ckpt_ocrmodel/model.ckpt.meta")
    print("Loading saved model...")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path)
        #x = sess.graph.get_tensor_by_name("data:0")
        input_img = sess.graph.get_tensor_by_name("input:0")  #输入节点
        wordlist = sess.graph.get_tensor_by_name("CTCBeamSearchDecoder:1") #输出节点

        #prob = sess.graph.get_tensor_by_name("prob:0")
        #values, indices = tf.nn.top_k(prob, 3)
        print(input_img)
        print(wordlist)

        export_path = '/data/liuyan/ocr_model/tf-wpai-serving-ocrmodel4'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_input_img = tf.saved_model.utils.build_tensor_info(input_img)
        tensor_info_list = tf.saved_model.utils.build_tensor_info(wordlist)

        #tensor_info_pro = tf.saved_model.utils.build_tensor_info(tf.reshape(values, [3]))
        #tensor_info_classify = tf.saved_model.utils.build_tensor_info(tf.reshape(indices,[3]))
        signature_def_map = {
            "ocr_image": tf.saved_model.signature_def_utils.build_signature_def(
                 inputs={"image": tensor_info_input_img},
                 outputs={
                     "wordlist": tensor_info_list,

                 },
                 method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
             )}
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map=signature_def_map)
        builder.save()
        print('builder.save finished.')


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
        input_img = sess.graph.get_tensor_by_name("image_tensor:0")
        classes = sess.graph.get_tensor_by_name("detection_classes:0")
        boxes = sess.graph.get_tensor_by_name("detection_boxes:0")
        scores = sess.graph.get_tensor_by_name("detection_scores:0")
        #prob = sess.graph.get_tensor_by_name("prob:0")
        #values, indices = tf.nn.top_k(prob, 3)
        print(input_img)
        print(classes)
        print(boxes)
        print(scores)
        export_path = '/data/liuyan/ocr_model/tf-wpai-serving-driving-ssd'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_input_img = tf.saved_model.utils.build_tensor_info(input_img)
        tensor_info_classes = tf.saved_model.utils.build_tensor_info(classes)
        tensor_info_boxes = tf.saved_model.utils.build_tensor_info(boxes)
        tensor_info_scores = tf.saved_model.utils.build_tensor_info(scores)
        #tensor_info_pro = tf.saved_model.utils.build_tensor_info(tf.reshape(values, [3]))
        #tensor_info_classify = tf.saved_model.utils.build_tensor_info(tf.reshape(indices,[3]))
        signature_def_map = {
            "drive_predict_image": tf.saved_model.signature_def_utils.build_signature_def(
                 inputs={"image": tensor_info_input_img},
                 outputs={
                     "class": tensor_info_classes,
                     "box": tensor_info_boxes,
                     "score": tensor_info_scores
                 },
                 method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
             )}
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map=signature_def_map)
        builder.save()
        print('builder.save finished.')

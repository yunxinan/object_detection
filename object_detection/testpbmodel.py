# coding=UTF-8

import tensorflow as tf
import matplotlib.image as mpimg
import time as tm
import numpy as np

def load_graph(model_dir):
    with tf.gfile.GFile(model_dir, "rb") as f: #读取模型数据
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) #得到模型中的计算图和数据

        with tf.Graph().as_default() as graph: #这里的Graph()要有括号，不然会报TypeError
            tf.import_graph_def(graph_def, name="") #导入模型中的图到现在这个新的计算图中，不指定名字的话默认是 import
            return graph


if __name__  == "__main__":
    graph = load_graph("/data/liuyan/car_plate/object_detection_pb/ssd_frozen_inference_graph/frozen_inference_graph.pb") #这里传入的是完整的路径包括pb的名字，不然会报FailedPreconditionError
    #f = open("/home/data/liuyan/out.txt", "w+", encoding="utf-8")

    for op in graph.get_operations(): #打印出图中的节点信息
        print (op.name, op.values())
        #print(op.name, file=f)   #将节点的名称打印在a.txt中

    x = graph.get_tensor_by_name('image_tensor:0') #得到输入节点tensor的名字，记得跟上导入图时指定的name
    y1 = graph.get_tensor_by_name('detection_classes:0') #得到输出节点tensor的名字
    y2 = graph.get_tensor_by_name('detection_boxes:0')
    y3 = graph.get_tensor_by_name('detection_scores:0')



    with tf.Session(graph=graph) as sess: #创建会话运行计算
        #img = mpimg.imread('/home/data/liuyan/000001.jpg')  #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        img = mpimg.imread('/data/liuyan/car_plate/VOCtest_Plate/JPEGImages/5113{split}5112.jpg')  # /home/data/liuyan/imagedata/VOCtest_06-Nov-2007/JPEGImages/000625.jpg
        #image_bilinear = tf.image.resize_images(img, size=[300, 300], method=tf.image.ResizeMethod.BILINEAR)
        img_4d = img[None]
        print(img_4d.shape)
        # img_4dtran=np.reshape(img_4d,(1,3,300,300))
        #img_4dtran = tf.transpose(img_4d, [0, 3, 1, 2])
        #img_f = tf.image.convert_image_dtype(img_4dtran, dtype=tf.float32)
        #img_NCHW = img_f.eval()
        #img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        """
        y_out1 = sess.run(y1, feed_dict={x:img_NCHW})
        print(y_out1)
        y_out2 = sess.run(y2, feed_dict={x:img_NCHW})
        print(y_out2)
        """
        t1 = tm.time()
        y_out = sess.run([y1,y2,y3], feed_dict={x: img_4d})
        print(y_out[0])
        print(y_out[1])
        print(y_out[2])
        print(y_out[0][0,0:1])
        print(y_out[1][0,0:1])
        print(y_out[2][0,0:1])
        #y_out['detection_classes'] = y_out ['detection_classes'][0].astype(np.uint8)

        #y_out['detection_boxes'] =  y_out['detection_boxes'][0]

        #y_out['detection_scores'] = y_out ['detection_scores'][0]

       # print(y_out['detection_classes'][0])
        #print(y_out['detection_boxes'][0])
        #print(y_out['detection_scores'][0])
        t2 = tm.time()
        print("==========total take {0}", format(t2 - t1))
        #print(y_out)

    print ("finish")




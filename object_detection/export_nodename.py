import tensorflow as tf

# from ckpt output ops name
saver = tf.train.import_meta_graph("/data/liuyan/identitycard/object_detection_ssd_model_6/model.ckpt-40000.meta")
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	saver.restore(sess, "/data/liuyan/identitycard/object_detection_ssd_model_6/model.ckpt-40000")
	f=open("/data/liuyan/a.txt","w+",encoding="utf-8")

    # Check all operations (nodes) in the graph:
    #print("## All operations: ")
	#f=open("/home/data/liuyan","w+",encoding="utf-8")
	for op in graph.get_operations():
		print(op.name,file=f)
		#fout = open("/home/data/liuyan/a.txt", "w+", encoding="utf-8")
		#fout.write("this string will be output in txt.")
import tensorflow as tf

"""tf.Graph()构造一个空白的新的图
tf.as_default()可以用这个方法访问其上下文,为期添加Op,结合with语句,
可以通过上下文管理器通知Tensorflow我们需要添加一些合适的O品添加到某个特定的Graph对象中
对于一个数据流模型而言,当tensorflow本加载时会自动的创建一个Graph对象,并将其作为默认的数据流图
而在Graph_default()上下文管理器之外的定义的任何Op,Tensor对象都会自动放置在默认的数据流图中"""
g1 = tf.Graph()
g2 = tf.Graph()
# 定义G1的Op和张量
with g1.as_default():
    in_graph_g1 = tf.multiply(2, 3)
# 定义G2的Op和张量
with g2.as_default():
    in_graph_g2 = tf.subtract(2, 3)
# 由于不在with语句块中被定义,下面的Op将会放置在默认的数据流图中
in_default_graph = tf.add(1, 2)

# 如果希望得到默认数据流图的句柄,可以使用tf.get_default_graph函数
default_graph = tf.get_default_graph()

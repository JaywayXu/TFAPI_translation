# 利用占位节点添加输入
"""占位符的意义是在创建时无需为他们指定具体的数值,
作用是为了在运行时即将到来的某个Tensor对象预留位置,因此实际上变成了"输入",
利用tf.placeholder创建占位符
"""
import tensorflow as tf
import numpy as np

# 创建一个长度为2,数据类型为int32的占位符向量
a = tf.placeholder(tf.int32, shape=[2], name="my_input")

# 将该占位向量视为其他任意Tensor对象,加以使用
b = tf.reduce_prod(a, name="prod_b")  # 指定维度上求乘积
c = tf.reduce_sum(a, name="sum_c")  # 指定维度上求和

# 完成数据流图的定义
d = tf.add(b, c, name="add_d")

"""为了给占位符传入一个实际的值,需要使用Session.run()中的feed_dict参数,
我们将以指向占位符输出的句柄作为字典的键,将传入tensor对象作为字典的值"""
# 定义一个tensorflow session对象
sess = tf.Session()
# 创建一个将床给feed_dict参数的字典
input_dict = {a: np.array([5, 3], dtype=np.int32)}

# 计算d的值,将input_dict的"值"传给a
print(sess.run(d, feed_dict=input_dict))

"""placeholder的值是无法计算的--如果试图将其传入Session.run(),将会引发一个异常"""

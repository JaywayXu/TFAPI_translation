import tensorflow as tf

"""tf.cos(x,name=None)
功能：计算x的余弦值。
输入：x为张量，可以为`half`,`float32`, `float64`,  `complex64`, `complex128`类型。"""
x = tf.constant([[0, 3.1415926]], tf.float64)
z = tf.cos(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# [[ 1. -1.]]

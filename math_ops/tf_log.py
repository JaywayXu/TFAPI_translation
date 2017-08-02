import tensorflow as tf

"""tf.log(x,name=None)
功能：计算x各元素的自然对数。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""

x = tf.constant([[1, 2.71828183, 10]], tf.float64)
z = tf.log(x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 1. 2.30258509]]
"""tf.log1p(x,name=None)
功能：计算x各元素加1后的自然对数。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""

x = tf.constant([[0, 1.71828183, 9]], tf.float64)
z = tf.log1p(x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 1. 2.30258509]]

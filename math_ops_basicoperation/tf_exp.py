import tensorflow as tf

"""tf.exp(x,name=None)
功能：计算x各元素的自然指数，即e^x。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""

x = tf.constant([[0, 1, -1]], tf.float64)
z = tf.exp(x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1. 2.71828183 0.36787944]
"""tf.expm1(x,name=None)
功能：计算x各元素的自然指数减1，即e^x-1。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""
x = tf.constant([[0, 1, -1]], tf.float64)
z = tf.expm1(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 1.71828183 -0.63212056]]

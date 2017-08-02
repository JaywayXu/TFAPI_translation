import tensorflow as tf

"""tf.sqrt(x,name=None)
功能：计算x各元素的平方根。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""

x = tf.constant([[2, 3, -5]], tf.float64)
z = tf.sqrt(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1.41421356 1.73205081 nan]]

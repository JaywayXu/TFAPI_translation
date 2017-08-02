import tensorflow as tf

"""tf.tan(x,name=None)
功能：计算tan(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。"""

x = tf.constant([[0, 0.785398163]], tf.float64)
z = tf.tan(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 1.]]

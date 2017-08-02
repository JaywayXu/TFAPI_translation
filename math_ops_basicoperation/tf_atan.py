import tensorflow as tf

"""tf.atan(x,name=None)
功能：计算atan(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。"""
x = tf.constant([[0, 1, -1]], tf.float64)
z = tf.atan(x)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[0. 0.78539816 -0.78539816]]

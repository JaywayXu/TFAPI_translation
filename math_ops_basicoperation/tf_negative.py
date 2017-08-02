import tensorflow as tf

"""tf.negative(x,name=None)
功能：求x的负数。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。"""

x = tf.constant([[1.1, 2, -3]], tf.float64)
z = tf.negative(x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[-1.1. -2. 3.]]

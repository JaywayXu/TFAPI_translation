import tensorflow as tf

"""tf.square(x,name=None)
功能：计算x各元素的平方。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。"""

x = tf.constant([[2, 0, -3]], tf.float64)
z = tf.square(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[4. 0. 9.]]

import tensorflow as tf

"""tf.reciprocal(x,name=None)
功能：求x的倒数。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。"""

x = tf.constant([[2, 0, -3]], tf.float64)
z = tf.reciprocal(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0.5 inf -0.33333333]]

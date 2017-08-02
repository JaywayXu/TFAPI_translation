import tensorflow as tf

"""tf.ceil(x,name=None)
功能：计算x各元素比x大的最小整数。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[0.2, 0.8, -0.7]], tf.float64)
z = tf.ceil(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1. 1. -0.]]

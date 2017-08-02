import tensorflow as tf

"""tf.rint(x,name=None)
功能：计算离x最近的整数，若为中间值，取偶数值。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[-1.7, -1.5, -1.1, 0.1, 0.5, 0.4, 1.5]], tf.float64)
z = tf.rint(x)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[-2. -2. -1. 0. 0. 0. 2.]]

import tensorflow as tf

"""tf.round(x,name=None)
功能：计算x各元素的距离其最近的整数，若在中间，则取偶数值。
输入：x为张量，可以为`float32`, `float64`类型。"""

x = tf.constant([[0.9, 1.1, 1.5, -4.1, -4.5, -4.9]], tf.float64)
z = tf.round(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1. 1. 2. -4. -4. -5.]]

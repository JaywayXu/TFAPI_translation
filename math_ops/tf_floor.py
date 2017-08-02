"""tf.floor(x,name=None)
功能：计算x各元素比其小的最大整数。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""
import tensorflow as tf

x = tf.constant([[0.2, 0.8, -0.7]], tf.float64)
z = tf.floor(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 0. -1.]]

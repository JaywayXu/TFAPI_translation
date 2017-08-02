import tensorflow as tf

"""tf.lgamma(x,name=None)
功能：计算ln(gamma(x))。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[1, 2, 3]], tf.float64)
z = tf.lgamma(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 0. 0.69314718]]
# gamma函数伽马函数gamma(x)=(n-1)!

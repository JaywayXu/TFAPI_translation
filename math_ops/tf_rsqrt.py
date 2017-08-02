import tensorflow as tf

"""tf.rsqrt(x,name=None)
功能：计算x各元素的平方根的倒数。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。"""

x = tf.constant([[2, 3, 5]], tf.float64)
z = tf.rsqrt(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0.70710678 0.57735027 0.4472136]]

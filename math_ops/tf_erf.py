import tensorflow as tf

"""tf.erf(x,name=None)
功能：计算x的高斯误差。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[-1, 0, 1, 2, 3]], tf.float64)
z = tf.erf(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[-0.84270079 0. 0.84270079 0.99532227 0.99997791]]

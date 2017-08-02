import tensorflow as tf

"""功能：计算Hurwitz zeta函数。
输入：x为张量或稀疏张量，可以为`float32`, `float64`类型。"""

a = tf.constant(1, tf.float64)
x = tf.constant([[1, 2, 3, 4]], tf.float64)
z = tf.zeta(x, a)
sess = tf.Session()
print(sess.run(z))
sess.close()
# [[        inf  1.64493407  1.2020569   1.08232323]]
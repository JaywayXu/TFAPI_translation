import tensorflow as tf

"""tf.matrix_determinant(input, name=None)
功能：求行列式。
输入：必须是float32，float64类型。"""

a = tf.constant([1, 2, 3, 4], shape=[2, 2], dtype=tf.float32)
z = tf.matrix_determinant(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>-2.0

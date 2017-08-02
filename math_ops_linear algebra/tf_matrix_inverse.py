import tensorflow as tf

"""tf.matrix_inverse(input, adjoint=None, name=None)
功能：求矩阵的逆。
输入：输入必须是float32，float64类型。adjoint表示计算前求转置 """

a = tf.constant([1, 2, 3, 4], shape=[2, 2], dtype=tf.float64)
z = tf.matrix_inverse(a)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[-2.    1.]
#      [1.5  -0.5]]

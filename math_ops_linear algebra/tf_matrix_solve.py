import tensorflow as tf

"""tf.matrix_solve(matrix, rhs, adjoint=None, name=None)
功能：求线性方程组，matrix*X=rhs。
输入：adjoint:是否对matrix转置。"""

a = tf.constant([2, -2, -2, 5], shape=[2, 2], dtype=tf.float64)
RHS = tf.constant([3, 10], shape=[2, 1], dtype=tf.float64)
z = tf.matrix_solve(a, RHS)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[5.83333333]
#      [4.33333333]]

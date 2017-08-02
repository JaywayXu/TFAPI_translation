import tensorflow as tf

"""tf.matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)
功能：求解多个线性方程的最小二乘问题。
输入：。"""
a = tf.constant([2, 4, -2, 5], shape=[2, 2], dtype=tf.float64)
RHS = tf.constant([3, 10], shape=[2, 1], dtype=tf.float64)
z = tf.matrix_solve_ls(a, RHS)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[-1.38888889]
#      [1.44444444]]

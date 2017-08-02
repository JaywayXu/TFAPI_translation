import tensorflow as tf

"""tf.cholesky_solve(chol, rhs, name=None)
功能：对方程‘AX=RHS’进行cholesky求解。
输入：chol=tf.cholesky(A)。"""

a = tf.constant([2, -2, -2, 5], shape=[2, 2], dtype=tf.float64)
chol = tf.cholesky(a)
RHS = tf.constant([3, 10], shape=[2, 1], dtype=tf.float64)
z = tf.cholesky_solve(chol, RHS)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[5.83333333]
#      [4.33333333]] #A*X=RHS

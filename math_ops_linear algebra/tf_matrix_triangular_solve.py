import tensorflow as tf

"""tf.matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None)
功能：求解matrix×X=rhs，matrix为上三角或下三角阵。
输入：lower：默认为None，matrix上三角元素为0;若为True，matrix下三角元素为0;
     adjoint：转置"""

a = tf.constant([2, 4, -2, 5], shape=[2, 2], dtype=tf.float64)
RHS = tf.constant([3, 10], shape=[2, 1], dtype=tf.float64)
z = tf.matrix_triangular_solve(a, RHS)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[1.5]
#      [2.6]]  这和lower=True设置的结果一样，但是如果将其设置为lower=False的话，表示下三角元素为0

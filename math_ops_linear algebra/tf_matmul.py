import tensorflow as tf

"""tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
功能：矩阵乘法。配置后的矩阵a，b必须满足矩阵乘法对行列的要求。
输入：transpose_a,transpose_b:运算前是否转置;
      adjoint_a,adjoint_b:运算前进行共轭;
     a_is_sparse,b_is_sparse:a，b是否当作稀疏矩阵进行运算。"""

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
z = tf.matmul(a, b)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[ 58  64]
#      [139 154]]

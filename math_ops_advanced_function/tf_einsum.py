import tensorflow as tf

"""tf.einsum(equation, *inputs)
功能：通过equation进行矩阵乘法。
输入：equation：乘法算法定义。
# 矩阵乘
>>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]
# 点乘
>>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
# 向量乘
>>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
# 转置
>>> einsum('ij->ji', m)  # output[j,i] = m[i,j]
# 批量矩阵乘
>>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]"""
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
z = tf.einsum('ij,jk->ik', a, b)  # 矩阵乘
z1 = tf.einsum('ij,ij->ij', a, b)  # 向量乘，即对应位置相乘，此时两个矩阵的形式会完全一致
z2 = tf.einsum('ij,ij->', a, b)  # 矩阵的点积，又名为内积，返回的是对应位置相乘后相加的标量值

sess = tf.Session()

print(sess.run(z))
print(sess.run(z1))
print(sess.run(z2))

sess.close()
# [[19 22]
#  [43 50]]
# [[ 5 12]
#  [21 32]]
#   70

import tensorflow as tf

"""tf.tensordot(a, b, axes, name=None)
功能：同numpy.tensordot，根据axis计算点乘。
输入：axes=1或axes=[[1],[0]]，即为矩阵乘。"""

a = tf.constant([1, 2, 3, 4], shape=[2, 2], dtype=tf.float64)
b = tf.constant([1, 2, 3, 4], shape=[2, 2], dtype=tf.float64)
z = tf.tensordot(a, b, axes=[[1], [1]])   # 第一个矩阵的行乘上第二个矩阵的行
z1 = tf.tensordot(a, b, axes=[[1], [0]])  # 矩阵乘法第一个矩阵行乘第二个矩阵的列
z2 = tf.tensordot(a, b, axes=[[0], [1]])  # 第一个矩阵的列乘上第二个矩阵的行
sess = tf.Session()
print(sess.run(z))
print(sess.run(z1))
print(sess.run(z2))
sess.close()
# z==>[[5.  11.]
#      [11. 25.]]
# z1==> [[  7.  10.]
#       [ 15.  22.]]
# z2==>[[  7.  15.]
#       [ 10.  22.]]

import tensorflow as tf

"""tf.matrix_set_diag(input,diagonal,name=None)
功能：将输入矩阵的对角元素置换为对角元素。
输入：input：矩阵，diagonal：对角元素。"""
a = tf.constant([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]])
z = tf.matrix_set_diag(a, [10, 11, 12, 13])
sess = tf.Session()
print(sess.run(z))
# z==>[[10  1  2  3]
#      [-1 11  1  2]
#      [0  -1 12  1]
#      [0   0 -1 13]]
# 当此矩阵不是对角矩阵时
a1 = tf.constant([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1]])
z1 = tf.matrix_set_diag(a1, [10, 11, 12])


print(sess.run(z1))
sess.close()

# [[10  1  2  3]
#  [-1 11  1  2]
#  [-2 -1 12  1]]

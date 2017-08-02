import tensorflow as tf

"""tf.matrix_diag_part(input,name=None)
功能：返回批对角阵的对角元素
输入：tensor,批对角阵"""

a = tf.constant([[[1, 3, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 5, 0], [0, 0, 6]]])
z = tf.matrix_diag_part(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1 2 3]
#      [4 5 6]]

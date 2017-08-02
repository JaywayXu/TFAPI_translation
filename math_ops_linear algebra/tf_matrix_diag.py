import tensorflow as tf

"""tf.matrix_diag(diagonal,name=None)
功能：根据对角值返回一批对角阵
输入：对角值"""
a = tf.constant([[1, 2, 3], [4, 5, 6]])
z = tf.matrix_diag(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[[1 0 0]
#       [0 2 0]
#       [0 0 3]]
#      [[4 0 0]
#       [0 5 0]
#       [0 0 6]]]

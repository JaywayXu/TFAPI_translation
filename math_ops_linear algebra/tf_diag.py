import tensorflow as tf

"""tf.diag(diagonal, name=None)
功能：返回对角阵。
输入：tensor，秩为k<=3。"""
a = tf.constant([1, 2, 3, 4])
z = tf.diag(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1 0 0 0
#       0 2 0 0
#       0 0 3 0
#       0 0 0 4]]

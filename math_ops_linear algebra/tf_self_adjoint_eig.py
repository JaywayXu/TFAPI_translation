import tensorflow as tf

"""tf.self_adjoint_eig(tensor, name=None)
功能：求取特征值和特征向量。"""

a = tf.constant([3, -1, -1, 3], shape=[2, 2], dtype=tf.float64)

sess = tf.Session()
print(sess.run(tf.self_adjoint_eig(a)))
sess.close()

# e==>[2.  4.]
# v==>[[0.70710678 0.70710678]
#     [0.70710678 -0.70710678]]

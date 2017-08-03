import tensorflow as tf

"""tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)
功能：进行奇异值分解。tensor=u×diag（s）×transpose（v）"""

a = tf.constant([3, -1, -1, 3], shape=[2, 2], dtype=tf.float64)
sess = tf.Session()
print(sess.run(tf.svd(a)))
sess.close()

# s==>[4. 2.]
# u==>[[0.70710678 0.70710678]
#     [-0.70710678 0.70710678]]
# v==>[[0.70710678 0.70710678]
# #     [-0.70710678 0.70710678]]

# (array([ 4.,  2.]), array([[ 0.70710678,  0.70710678],
#        [-0.70710678,  0.70710678]]), array([[ 0.70710678,  0.70710678],
#        [-0.70710678,  0.70710678]]))

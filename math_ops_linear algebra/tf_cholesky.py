import tensorflow as tf

tf.cholesky(input, name=None)
"""功能：进行cholesky分解。
输入：注意输入必须是正定矩阵。"""
a = tf.constant([2, -2, -2, 5], shape=[2, 2], dtype=tf.float64)
z = tf.cholesky(a)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[ 1.41421356  0.        ]
#      [-1.41421356  1.73205081]]

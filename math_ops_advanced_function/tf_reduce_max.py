import tensorflow as tf

"""tf.reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算最大值，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[1, 2, 3], [4, 5, 6]])
z = tf.reduce_max(a)
z2 = tf.reduce_max(a, 0)
z3 = tf.reduce_max(a, 1)
z4 = tf.reduce_max(a, 1, keep_dims=True)
sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()


# z==>6
# z2==>[4 5 6]
# z3==>[3 6]
# z4==>[[3] [6]]

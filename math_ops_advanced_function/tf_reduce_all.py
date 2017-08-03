import tensorflow as tf

"""tf.reduce_all(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算逻辑与，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[True, True, False, False], [True, False, False, True]])
z = tf.reduce_all(a)
z2 = tf.reduce_all(a, 0)
z3 = tf.reduce_all(a, 1)

sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))

sess.close()


# z==>False
# z2==>[True False False False]
# z3==>[False False]

import tensorflow as tf

"""tf.count_nonzero(input_tensor, axis=None, keep_dims=False, dtype=tf.int64, name=None, reduction_indices=None)
功能：沿着维度axis计算非0个数，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[0, 0, 0], [0, 1, 2]], dtype=tf.float64)
z = tf.count_nonzero(a)
z2 = tf.count_nonzero(a, 0)
z3 = tf.count_nonzero(a, 1)

sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
sess.close()

# z==>2
# z2==>[0 1 1]
# z3==>[0 2]

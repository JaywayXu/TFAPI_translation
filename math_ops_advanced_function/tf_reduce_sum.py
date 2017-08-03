import tensorflow as tf

"""tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算元素和，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[1, 2, 3], [4, 5, 6]])
z = tf.reduce_sum(a)  # 所有维度求和
z2 = tf.reduce_sum(a, 0)  # 列维度求和
z3 = tf.reduce_sum(a, 1)  # 行维度求和
z4 = tf.reduce_sum(a, 1, keep_dims=True)

sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()

# z==>21
# z2==>[5 7 9]
# z3==>[6 15]
# z4==>[[ 6]
#       [15]]

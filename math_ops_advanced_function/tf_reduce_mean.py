import tensorflow as tf

"""tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算平均值，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float64)
z = tf.reduce_mean(a)
z2 = tf.reduce_mean(a, 0)
z3 = tf.reduce_mean(a, 1)
z4 = tf.reduce_mean(a, 1, keep_dims=True)
sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()

# z==>3.5
# z2==>[2.5 3.5 4.5]
# z3==>[2. 5.]
# z4==>[[ 2.]
#      [ 5.]]
